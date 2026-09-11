"""
Lightweight, opt-in scaling regression tests.

These check a few qualitative scaling *properties* with small, fast problem
sizes and generous tolerances -- they are regression guards against a
change accidentally reintroducing quadratic-or-worse behavior somewhere it
shouldn't be, not a precise complexity study. For that, see
`benchmarks/scaling_benchmark.py`, which is a standalone script (not
collected by pytest) that sweeps realistic problem sizes and reports
empirical scaling exponents.

Skipped by default: wall-clock-based tests are inherently more sensitive to
machine load than the rest of the suite, and this repo's CI runs the full
test suite across 15 OS/Python combinations (see .github/workflows/Build.yml)
where that noise would be amplified. Opt in explicitly:

    RUN_SCALING_TESTS=1 pytest tests/test_scaling.py -v
"""

import os
import time

import numpy as np
import pytest

from scaled_vecchia import ScaledVecchiaGP

if os.environ.get("RUN_SCALING_TESTS") != "1":
    pytest.skip("scaling tests are opt-in: set RUN_SCALING_TESTS=1 to run them "
                "(they are wall-clock-based and not run by default/in CI)",
                allow_module_level=True)


def _data(n, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sin(3 * X[:, 0]) + 0.1 * X[:, 1]
    return X, y


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def test_fit_time_plateaus_once_n_exceeds_n_est():
    """fit() always subsamples to at most n_est points, so time at n = 4x
    n_est should not be dramatically larger than at n = n_est."""
    n_est = 400
    X_small, y_small = _data(n_est)
    X_large, y_large = _data(4 * n_est)

    gp = ScaledVecchiaGP(m_est=10, n_est=n_est, random_state=0, var_correction=False)
    _time(gp.fit, X_small, y_small)   # warm up JIT before the measured calls
    t_small = _time(gp.fit, X_small, y_small)
    t_large = _time(gp.fit, X_large, y_large)

    assert t_large < 3.0 * t_small, (
        f"fit() at n=4*n_est ({t_large:.3f}s) should be close to fit() at "
        f"n=n_est ({t_small:.3f}s), not much larger -- both use at most "
        f"n_est={n_est} points internally"
    )


def test_fit_time_does_not_grow_worse_than_quadratically_in_n_est():
    """fit() is documented as O(n_est) (plus a smaller O(n_est^2) ordering
    term -- see benchmarks/scaling_benchmark.py's fit_n_est sweep for the
    full picture); this just guards against a regression to something
    worse, with a generous margin above the theoretical O(n_est^2) bound."""
    n0 = 300
    X0, y0 = _data(n0)
    X1, y1 = _data(4 * n0)

    gp0 = ScaledVecchiaGP(m_est=10, n_est=n0, random_state=0, var_correction=False)
    _time(gp0.fit, X0, y0)   # warm up
    t0 = _time(gp0.fit, X0, y0)

    gp1 = ScaledVecchiaGP(m_est=10, n_est=4 * n0, random_state=0, var_correction=False)
    t1 = _time(gp1.fit, X1, y1)

    # 4x n_est: allow up to ~24x time (well above the O(n_est^2)=16x bound)
    assert t1 < 24.0 * t0, (
        f"fit() time grew {t1 / t0:.1f}x for a 4x increase in n_est "
        f"({t0:.3f}s -> {t1:.3f}s); expected roughly linear-ish growth, "
        f"well under the O(n_est^2)=16x bound"
    )


def test_prepare_joint_sample_much_cheaper_per_call_than_naive_repeats():
    """The MCMC-loop use case: prepare_joint() once + repeated .sample()
    must be substantially cheaper per call than repeated sample_joint()
    (which redoes the full setup every time). Measured ~6-20x in practice
    (see README's "Repeated prediction" section); this only guards against
    a regression, so the bar is set low."""
    X, y = _data(600)
    gp = ScaledVecchiaGP(m_est=15, m_pred=25, n_est=600, random_state=0,
                          var_correction=False).fit(X, y)
    Xte, _ = _data(30, seed=1)
    n_calls = 20

    gp.sample_joint(Xte, n_sim=1, m=25)   # warm up

    t0 = time.perf_counter()
    for _ in range(n_calls):
        gp.sample_joint(Xte, n_sim=1, m=25)
    t_naive = time.perf_counter() - t0

    cache = gp.prepare_joint(Xte, m=25)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        cache.sample(n_sim=1)
    t_cached = time.perf_counter() - t0

    assert t_cached * 3.0 < t_naive, (
        f"prepare_joint()+.sample() ({t_cached:.4f}s for {n_calls} calls) "
        f"should be well under a third of naive repeated sample_joint() "
        f"({t_naive:.4f}s) -- got only {t_naive / t_cached:.1f}x speedup"
    )


def test_predict_time_grows_much_more_slowly_than_joint_setup_in_n_train():
    """predict() (marginal, cKDTree-based) should scale far more gently with
    training-set size than prepare_joint()'s exact O(n^2) maximin ordering
    -- guards against, e.g., accidentally routing predict() through the
    O(n^2) path."""
    n0, n1 = 500, 4000
    Xte, _ = _data(30, seed=2)

    X0, y0 = _data(n0)
    gp0 = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=n0, random_state=0,
                           var_correction=False).fit(X0, y0)
    gp0.predict(Xte)   # warm up
    t_predict_0 = _time(gp0.predict, Xte)
    t_joint_0 = _time(gp0.prepare_joint, Xte, m=20)

    X1, y1 = _data(n1)
    gp1 = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=n1, random_state=0,
                           var_correction=False).fit(X1, y1)
    t_predict_1 = _time(gp1.predict, Xte)
    t_joint_1 = _time(gp1.prepare_joint, Xte, m=20)

    predict_growth = t_predict_1 / max(t_predict_0, 1e-6)
    joint_growth = t_joint_1 / max(t_joint_0, 1e-6)
    assert predict_growth < joint_growth, (
        f"predict() time grew {predict_growth:.1f}x and prepare_joint() grew "
        f"{joint_growth:.1f}x for the same {n1 / n0:.0f}x increase in "
        f"training size; expected predict() to grow more slowly"
    )
