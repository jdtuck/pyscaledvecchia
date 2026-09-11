"""
Scaling benchmark for `ScaledVecchiaGP.fit` and its `predict`-family methods.

Purpose
-------
The README documents the paper's asymptotic cost -- O(n_est * m_est^3) for
fitting, O(n* * m_pred^3) for (marginal) prediction -- and separately notes
that the maximin ordering used by `predict_joint`/`sample_joint`/
`prepare_joint` is still the exact O(n^2) algorithm. This script empirically
measures wall-clock scaling for each of these regimes so the numbers can be
checked against those claims, compared across machines/branches, or used to
pick `n_est`/`m_est`/`m_pred` for a given problem size and time budget.

It is a script, not a pytest test (nothing here is collected by `pytest`,
per `testpaths = ["tests"]` in pyproject.toml) -- run it directly:

    python benchmarks/scaling_benchmark.py --sweep all
    python benchmarks/scaling_benchmark.py --sweep fit_n_est --plot --csv results.csv
    python benchmarks/scaling_benchmark.py --quick   # small sizes, fast smoke run

For a small, fast, opt-in pytest-based regression guard on a few of the
qualitative properties measured here (e.g. "fit time plateaus once n exceeds
n_est", "prepare_joint + .sample() is much cheaper per call than repeated
sample_joint calls"), see tests/test_scaling.py.

Sweeps
------
- fit_n_est          : fit() time vs n_est, with n >> max(n_est) fixed
                        (isolates the O(n_est) term; n itself shouldn't matter)
- fit_n_plateau      : fit() time vs n, holding n_est fixed -- demonstrates
                        the plateau once n exceeds n_est (fitting always
                        subsamples to at most n_est points)
- fit_m_est          : fit() time vs m_est, holding n_est fixed -- probes the
                        O(m_est^3) per-block term
- predict_n_test     : marginal predict() time vs number of query points,
                        holding the training set fixed -- expected ~linear
- predict_n_train    : marginal predict() time vs training set size, holding
                        the query set fixed -- expected mild (~n log n) growth
                        from the cKDTree build, much milder than joint setup
- joint_setup_n_train: prepare_joint()'s one-time setup cost vs training set
                        size -- expected ~quadratic (the exact maximin
                        ordering over the combined train+test set); this is
                        the cost `prepare_joint`/order_obs caching amortizes
                        away across repeated calls, not the cost itself
- repeated_sampling  : head-to-head comparison of naive repeated
                        `sample_joint` calls vs `prepare_joint()` once +
                        repeated `.sample()`, reporting the per-call speedup
                        (the MCMC-loop use case)

Each sweep prints a table and, for the "vs size" sweeps, fits an approximate
power-law exponent via log-log linear regression (time ~ size^exponent) so
the empirical complexity is a single comparable number, not just a table to
eyeball.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

from scaled_vecchia import ScaledVecchiaGP


# ---------------------------------------------------------------------------
# Synthetic data and timing helpers
# ---------------------------------------------------------------------------

def make_data(n, d, seed=0, noise=0.05):
    """A smooth, cheap-to-evaluate d-dimensional function -- only the first
    few dimensions matter, mirroring the kind of "effective low-dimensionality"
    computer-model emulation is often applied to (see examples/borehole_demo.py
    for a closer-to-real example)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    active = min(d, 3)
    y = np.sin(3 * X[:, 0])
    if active > 1:
        y = y + 0.3 * np.cos(2 * X[:, 1])
    if active > 2:
        y = y + 0.1 * X[:, 2]
    y = y + noise * rng.standard_normal(n)
    return X, y


def timeit(fn, *args, reps=1, **kwargs):
    """Median wall-clock time over `reps` repetitions, plus the last result."""
    times = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def warmup():
    """Triggers one-time JIT compilation (Numba kernels, and the optional
    Cython extension's first import/use) on a throwaway tiny problem before
    any timed measurement, so it doesn't contaminate the first data point of
    whichever sweep happens to run first -- Numba's @njit compile alone can
    take longer than an entire subsequent real measurement at small sizes.
    """
    print("[warming up JIT-compiled kernels before timing...]", end=" ", flush=True)
    t0 = time.perf_counter()
    X, y = make_data(60, 3, seed=0)
    gp = ScaledVecchiaGP(m_est=10, m_pred=10, n_est=60, random_state=0,
                          var_correction=False).fit(X, y)
    Xte, _ = make_data(10, 3, seed=1)
    gp.predict(Xte)
    gp.prepare_joint(Xte, m=10).sample(n_sim=1)
    print(f"done ({time.perf_counter() - t0:.1f}s)")


def power_law_exponent(sizes, times):
    """Least-squares slope of log(time) vs log(size): time ~ size^slope."""
    sizes = np.asarray(sizes, dtype=float)
    times = np.asarray(times, dtype=float)
    mask = (sizes > 0) & (times > 0)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(sizes[mask]), np.log(times[mask]), 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

class Results:
    """Accumulates rows across all sweeps for optional CSV export/plotting."""

    def __init__(self):
        self.rows = []   # each: dict(sweep=..., x_label=..., x=..., metric=..., seconds=...)

    def add(self, sweep, x_label, x, metric, seconds):
        self.rows.append(dict(sweep=sweep, x_label=x_label, x=x,
                               metric=metric, seconds=seconds))

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sweep", "x_label", "x", "metric", "seconds"])
            w.writeheader()
            w.writerows(self.rows)
        print(f"\n[wrote {len(self.rows)} rows to {path}]")

    def plot(self, out_dir="."):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[--plot requested but matplotlib is not installed; skipping. "
                  "Install with: pip install -e '.[demo]']")
            return
        sweeps = sorted({r["sweep"] for r in self.rows})
        for sweep in sweeps:
            rows = [r for r in self.rows if r["sweep"] == sweep]
            metrics = sorted({r["metric"] for r in rows})
            fig, ax = plt.subplots(figsize=(5, 4))
            for metric in metrics:
                xs = [r["x"] for r in rows if r["metric"] == metric]
                ys = [r["seconds"] for r in rows if r["metric"] == metric]
                ax.loglog(xs, ys, marker="o", label=metric)
            ax.set_xlabel(rows[0]["x_label"])
            ax.set_ylabel("wall time (s)")
            ax.set_title(sweep)
            if len(metrics) > 1:
                ax.legend()
            fig.tight_layout()
            path = f"{out_dir}/scaling_{sweep}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            print(f"[wrote {path}]")


def print_table(title, headers, rows):
    print(f"\n--- {title} ---")
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep_fit_n_est(results, n_ests, d, n_fixed, m_est, reps, seed=0):
    """fit() time vs n_est, with n fixed well above max(n_ests): isolates the
    O(n_est) term, since fitting always subsamples to at most n_est points."""
    X, y = make_data(n_fixed, d, seed=seed)
    rows = []
    for n_est in n_ests:
        gp = ScaledVecchiaGP(m_est=m_est, n_est=n_est, random_state=seed,
                              var_correction=False)
        t, _ = timeit(gp.fit, X, y, reps=reps)
        rows.append((n_est, f"{t:.3f}"))
        results.add("fit_n_est", "n_est", n_est, "fit", t)
    print_table(f"fit() vs n_est  (n={n_fixed} fixed, m_est={m_est})",
                ["n_est", "fit (s)"], rows)
    exp = power_law_exponent(n_ests, [r["seconds"] for r in results.rows
                                       if r["sweep"] == "fit_n_est"])
    print(f"empirical exponent (time ~ n_est^k): k = {exp:.2f}  "
          f"(paper's stated complexity: O(n_est), i.e. k ~ 1)")


def sweep_fit_n_plateau(results, ns, d, n_est, m_est, reps, seed=0):
    """fit() time vs n, holding n_est fixed: should plateau once n exceeds
    n_est, since fitting always works on a subsample of at most n_est points."""
    rows = []
    for n in ns:
        X, y = make_data(n, d, seed=seed)
        gp = ScaledVecchiaGP(m_est=m_est, n_est=n_est, random_state=seed,
                              var_correction=False)
        t, _ = timeit(gp.fit, X, y, reps=reps)
        rows.append((n, "yes" if n > n_est else "no", f"{t:.3f}"))
        results.add("fit_n_plateau", "n", n, "fit", t)
    print_table(f"fit() vs n  (n_est={n_est} fixed, m_est={m_est})",
                ["n", "n > n_est?", "fit (s)"], rows)
    below = [r for r in rows if r[1] == "no"]
    above = [r for r in rows if r[1] == "yes"]
    if below and above:
        print(f"below n_est: {below[0][2]}s -> {below[-1][2]}s  |  "
              f"above n_est (should be roughly flat): "
              f"{above[0][2]}s -> {above[-1][2]}s")


def sweep_fit_m_est(results, m_ests, d, n_est, reps, seed=0):
    """fit() time vs m_est, holding n_est fixed: probes the O(m_est^3)
    per-conditioning-set-block term."""
    X, y = make_data(max(n_est, 2000), d, seed=seed)
    rows = []
    for m_est in m_ests:
        gp = ScaledVecchiaGP(m_est=m_est, n_est=n_est, random_state=seed,
                              var_correction=False)
        t, _ = timeit(gp.fit, X, y, reps=reps)
        rows.append((m_est, f"{t:.3f}"))
        results.add("fit_m_est", "m_est", m_est, "fit", t)
    print_table(f"fit() vs m_est  (n_est={n_est} fixed)", ["m_est", "fit (s)"], rows)
    exp = power_law_exponent(m_ests, [r["seconds"] for r in results.rows
                                       if r["sweep"] == "fit_m_est"])
    print(f"empirical exponent (time ~ m_est^k): k = {exp:.2f}  "
          f"(paper's stated complexity: O(m_est^3), i.e. k ~ 3)")


def sweep_predict_n_test(results, n_tests, d, n_train, m_pred, reps, seed=0):
    """Marginal predict() time vs number of query points, fixed training set."""
    X, y = make_data(n_train, d, seed=seed)
    gp = ScaledVecchiaGP(m_pred=m_pred, n_est=n_train, random_state=seed,
                          var_correction=False).fit(X, y)
    rows = []
    for n_test in n_tests:
        Xte, _ = make_data(n_test, d, seed=seed + 1)
        t, _ = timeit(gp.predict, Xte, reps=reps)
        rows.append((n_test, f"{t:.4f}"))
        results.add("predict_n_test", "n_test", n_test, "predict", t)
    print_table(f"predict() vs n_test  (n_train={n_train}, m_pred={m_pred})",
                ["n_test", "predict (s)"], rows)
    exp = power_law_exponent(n_tests, [r["seconds"] for r in results.rows
                                        if r["sweep"] == "predict_n_test"])
    print(f"empirical exponent (time ~ n_test^k): k = {exp:.2f}  "
          f"(expected roughly linear, k ~ 1)")


def sweep_predict_n_train(results, ns_train, d, n_test, m_pred, reps, seed=0):
    """Marginal predict() time vs training set size, fixed query set. This
    only exercises the cKDTree build/query in _predict_scaled, NOT the
    O(n^2) maximin ordering that joint prediction pays -- expected much
    milder growth than joint_setup_n_train below."""
    Xte, _ = make_data(n_test, d, seed=seed + 1)
    rows = []
    for n_train in ns_train:
        X, y = make_data(n_train, d, seed=seed)
        gp = ScaledVecchiaGP(m_pred=m_pred, n_est=n_train, random_state=seed,
                              var_correction=False).fit(X, y)
        t, _ = timeit(gp.predict, Xte, reps=reps)
        rows.append((n_train, f"{t:.4f}"))
        results.add("predict_n_train", "n_train", n_train, "predict", t)
    print_table(f"predict() vs n_train  (n_test={n_test}, m_pred={m_pred})",
                ["n_train", "predict (s)"], rows)
    exp = power_law_exponent(ns_train, [r["seconds"] for r in results.rows
                                         if r["sweep"] == "predict_n_train"])
    print(f"empirical exponent (time ~ n_train^k): k = {exp:.2f}  "
          f"(expected mild, well under 2 -- contrast with joint_setup_n_train)")


def sweep_joint_setup_n_train(results, ns_train, d, n_test, m_pred, reps, seed=0):
    """prepare_joint()'s one-time setup cost vs training set size: exercises
    the exact O(n^2) maximin ordering over the combined train+test set
    (documented as a known limitation in the README). This is exactly the
    cost that order_obs caching / prepare_joint amortizes across repeated
    calls -- see sweep_repeated_sampling for that side of the story."""
    Xte, _ = make_data(n_test, d, seed=seed + 1)
    rows = []
    for n_train in ns_train:
        X, y = make_data(n_train, d, seed=seed)
        gp = ScaledVecchiaGP(m_pred=m_pred, n_est=n_train, random_state=seed,
                              var_correction=False).fit(X, y)
        t, _ = timeit(gp.prepare_joint, Xte, m=m_pred, reps=reps)
        rows.append((n_train, f"{t:.4f}"))
        results.add("joint_setup_n_train", "n_train", n_train, "prepare_joint", t)
    print_table(f"prepare_joint() setup vs n_train  (n_test={n_test}, m_pred={m_pred})",
                ["n_train", "prepare_joint (s)"], rows)
    exp = power_law_exponent(ns_train, [r["seconds"] for r in results.rows
                                         if r["sweep"] == "joint_setup_n_train"])
    print(f"empirical exponent (time ~ n_train^k): k = {exp:.2f}  "
          f"(expected close to 2: exact O(n^2) maximin ordering)")
    print("  note: at moderate n_train, fixed per-call overhead that does NOT "
          "scale with n_train (ordering/covariance work over the n_test side) "
          "can dilute the fitted exponent below the true asymptotic ~2; a "
          "wider/larger n_train range gives a cleaner read (isolating just "
          "`ordering.maximin_order(gp.X_ / gp.ranges_)` gives the cleanest "
          "signal of all, with no such confound).")


def sweep_repeated_sampling(results, n_train, n_test, d, m_pred, n_calls, seed=0):
    """The MCMC-loop use case: many single-sample draws at a *fixed* Xtest.
    Compares naive repeated sample_joint() calls (redoes the full setup
    every time) against prepare_joint() once + repeated .sample() (reuses
    the cached factorization)."""
    X, y = make_data(n_train, d, seed=seed)
    gp = ScaledVecchiaGP(m_pred=m_pred, n_est=n_train, random_state=seed,
                          var_correction=False).fit(X, y)
    Xte, _ = make_data(n_test, d, seed=seed + 1)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        gp.sample_joint(Xte, n_sim=1, m=m_pred)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    cache = gp.prepare_joint(Xte, m=m_pred)
    for _ in range(n_calls):
        cache.sample(n_sim=1)
    t_cached = time.perf_counter() - t0

    rows = [
        ("naive: sample_joint() x N", f"{t_naive:.3f}", f"{t_naive / n_calls * 1000:.3f}"),
        ("prepare_joint() + .sample() x N", f"{t_cached:.3f}", f"{t_cached / n_calls * 1000:.3f}"),
    ]
    print_table(f"repeated single-sample draws at fixed Xtest  "
                f"(n_train={n_train}, n_test={n_test}, N={n_calls} calls)",
                ["pattern", "total (s)", "ms/call"], rows)
    print(f"speedup: {t_naive / t_cached:.1f}x")
    results.add("repeated_sampling", "pattern", "naive", "total_s", t_naive)
    results.add("repeated_sampling", "pattern", "prepare_joint", "total_s", t_cached)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SWEEP_NAMES = ["fit_n_est", "fit_n_plateau", "fit_m_est", "predict_n_test",
               "predict_n_train", "joint_setup_n_train", "repeated_sampling"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", choices=SWEEP_NAMES + ["all"], default="all")
    p.add_argument("--d", type=int, default=4, help="input dimension")
    p.add_argument("--reps", type=int, default=1, help="repetitions per point (median reported)")
    p.add_argument("--plot", action="store_true", help="save log-log plots (requires matplotlib)")
    p.add_argument("--csv", type=str, default=None, help="write all raw results to this CSV path")
    p.add_argument("--quick", action="store_true",
                    help="small sizes for a fast smoke run, e.g. while developing this script")
    args = p.parse_args()

    if args.quick:
        n_ests = [200, 400, 800]
        ns_plateau = [200, 400, 1200, 2400]
        m_ests = [10, 15, 20]
        n_tests = [50, 100, 200]
        ns_train = [300, 600, 1200]
        n_calls = 30
        n_est_fixed, n_test_fixed, m_pred_fixed = 400, 50, 20
    else:
        n_ests = [500, 1000, 2000, 4000]
        ns_plateau = [1000, 2000, 6000, 12000]
        m_ests = [10, 20, 30, 45]
        n_tests = [100, 300, 1000, 3000]
        ns_train = [1000, 2000, 4000, 8000]
        n_calls = 200
        n_est_fixed, n_test_fixed, m_pred_fixed = 3000, 200, 60

    results = Results()
    sweeps = SWEEP_NAMES if args.sweep == "all" else [args.sweep]

    print(f"scaling_benchmark.py -- d={args.d}, reps={args.reps}, "
          f"quick={args.quick}, sweeps={sweeps}")
    warmup()

    if "fit_n_est" in sweeps:
        sweep_fit_n_est(results, n_ests, args.d, n_fixed=max(n_ests) * 3,
                         m_est=20, reps=args.reps)
    if "fit_n_plateau" in sweeps:
        sweep_fit_n_plateau(results, ns_plateau, args.d, n_est=n_est_fixed,
                             m_est=20, reps=args.reps)
    if "fit_m_est" in sweeps:
        sweep_fit_m_est(results, m_ests, args.d, n_est=n_est_fixed, reps=args.reps)
    if "predict_n_test" in sweeps:
        sweep_predict_n_test(results, n_tests, args.d, n_train=n_est_fixed,
                              m_pred=m_pred_fixed, reps=args.reps)
    if "predict_n_train" in sweeps:
        sweep_predict_n_train(results, ns_train, args.d, n_test=n_test_fixed,
                               m_pred=m_pred_fixed, reps=args.reps)
    if "joint_setup_n_train" in sweeps:
        sweep_joint_setup_n_train(results, ns_train, args.d, n_test=n_test_fixed,
                                   m_pred=m_pred_fixed, reps=args.reps)
    if "repeated_sampling" in sweeps:
        sweep_repeated_sampling(results, n_train=n_est_fixed, n_test=n_test_fixed,
                                 d=args.d, m_pred=m_pred_fixed, n_calls=n_calls)

    if args.csv:
        results.write_csv(args.csv)
    if args.plot:
        results.plot()


if __name__ == "__main__":
    sys.exit(main())
