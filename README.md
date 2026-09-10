[![Pipeline Status](https://github.com/jdtuck/pyscaledvecchia/actions/workflows/Build.yml/badge.svg)](https://github.com/jdtuck/pyscaledvecchia/actions/workflows/Build.yml)

# pyscaledvecchia

A NumPy/SciPy implementation of

> Katzfuss, M., Guinness, J., & Lawrence, E. (2020/2022). **"Scaled Vecchia
> approximation for fast computer-model emulation."** *SIAM/ASA Journal on
> Uncertainty Quantification*, 10(2). [arXiv:2005.00386](https://arxiv.org/abs/2005.00386)

The scaled Vecchia approximation fits an anisotropic (ARD) Gaussian-process
emulator to large computer experiments (n in the thousands to hundreds of
thousands) at a cost of `O(n m^3)` for fitting and `O(n* m*^3)` for
prediction, where `m` is a small conditioning-set size (default 30 for
fitting, 140 for prediction). It does this by:

1. **Scaling** each input dimension by its estimated range/length-scale
   parameter, `x~ = (x_1/lambda_1, ..., x_d/lambda_d)` (Sec. 3.1). This
   automatically performs a soft form of variable selection: irrelevant
   inputs get a very large `lambda_l` and stop influencing the neighbour
   structure.
2. **Ordering** the (scaled) inputs with an exact maximin ("farthest point")
   ordering, then building nearest-neighbour conditioning sets *in the scaled
   space* (Sec. 3.1).
3. **Maximizing** the resulting Vecchia log-likelihood with Fisher scoring,
   analytically profiling out the linear mean-function coefficients by GLS,
   and periodically refreshing the ordering/neighbours as the scaling
   estimate improves (iterations k = 2, 4, 8, 16, ...) (Sec. 3.2).
4. **Predicting** via an "observed-first" maximin ordering of the combined
   train + test inputs, which produces an (approximate) *joint* Gaussian
   predictive distribution with a sparse inverse-Cholesky factor — so you can
   draw correlated sample paths, not just independent marginals (Sec. 3.3).
5. Optionally **correcting predictive variances** by a scalar factor `b`
   estimated on a held-out inner split (Sec. 3.4).

The covariance function is an isotropic Matern plus a relative nugget:

```
K(x_i, x_j) = sigma^2 * [ M_nu(q_ij) + tau * 1{i == j} ]
q_ij        = sqrt( sum_l ((x_il - x_jl) / lambda_l)^2 )
```

`nu` (smoothness) can be:

- fixed at `0.5`, `1.5`, or `2.5` (the default is `2.5`) -- closed-form
  Matern, no Bessel function calls, and Numba-accelerated;
- fixed at any other positive value -- falls back to the general
  Bessel-based Matern (`scipy.special.kv`), which is not Numba-accelerated
  and so is slower per conditioning-set block (this path exploits the
  symmetry of the distance matrix to only evaluate the Bessel function on
  each block's K(K+1)/2 unique entries rather than all K^2, roughly a 2x
  saving over the naive approach, but it is still meaningfully slower than
  the closed-form nu in {0.5, 1.5, 2.5});
- or **estimated** by passing `nu=None`, following Sec. 3.5 of the paper and
  matching its own reference R implementation (`GpGp`/`GPvecchia`, which
  estimates nu jointly with the other covariance parameters via Fisher
  scoring when it isn't fixed by the user, starting from an initial
  smoothness of 3.5). This always uses the general Bessel-based path, since
  the derivative of a Bessel function with respect to its *order* has no
  closed form -- see `_matern_general.py` for exactly how the gradient and
  Fisher information for nu are computed (analytic where possible, a
  central finite difference only for the one component that genuinely has
  none). Note that `nu` is a famously hard covariance parameter to pin down
  precisely from finite data (see e.g. Zhang 2004 on infill asymptotics for
  the Matern family); treat point estimates of it accordingly.

All parameters (`sigma^2`, `lambda_1..d`, `nu` if estimated, `tau`) are
optimized on the log scale so positivity is automatic.

Pure NumPy/SciPy/Numba by default — no compiled extension is required (Numba
JIT-compiles the hot loops at runtime and caches them to disk). The general/
estimated-nu path additionally uses `scipy.special.kv`, and gets faster still
if the **optional** `scaled_vecchia._matern_cy` Cython/OpenMP extension was
built at install time:

- It fuses the whole per-block computation (distance, Bessel evaluation,
  analytic r-derivative, finite-difference nu-derivative, nugget) into one
  native loop calling `scipy.special.cython_special.kv` directly -- the
  *same* validated Bessel implementation `scipy.special.kv` uses, just
  reached at the C level with no per-call Python dispatch -- instead of the
  several separate NumPy array passes the pure-Python path needs. On its
  own this is a modest ~1.2-1.3x over the (already symmetry-optimized)
  pure-NumPy general-Matern path.
- More importantly, that Bessel call is `nogil`, so the loop over
  independent conditioning-set blocks runs under OpenMP across multiple
  threads (`ScaledVecchiaGP(n_jobs=...)`, default `-1` = all cores) --
  something the pure-NumPy path cannot do at all. This is where most of the
  real speedup comes from on a multi-core machine.
- This extension is entirely optional and additive: `pip install` attempts
  to build it, but degrades gracefully (falling back to pure NumPy/SciPy,
  with a build-time warning visible under `pip install -v`) if a C compiler
  or Cython isn't available, or compilation fails for any other reason --
  this never blocks installing the rest of the package, including the
  default (fixed nu in {0.5, 1.5, 2.5}, Numba-accelerated) fast path, which
  doesn't depend on it at all. Whether it's active is exposed as
  `scaled_vecchia.HAVE_CYTHON_BACKEND` and reported by `gp.summary()`.
  Because it delegates the actual Bessel-function math to SciPy's own
  tested implementation rather than reimplementing it, it carries no more
  numerical-correctness risk than the pure-Python fallback -- verified
  numerically identical in `tests/test_matern_cy.py` (which is skipped
  automatically if the extension wasn't built). OpenMP is enabled on Linux
  and Windows; on macOS the extension still builds and runs (single Bessel
  evaluations are still faster), just without multi-threading, since stock
  Apple Clang doesn't support `-fopenmp` out of the box.

## Installation

```bash
pip install -e .
```

or, to also pull in the test dependencies:

```bash
pip install -e ".[test]"
```

Requires Python >= 3.9, NumPy >= 1.22, SciPy >= 1.8, Numba >= 0.58. A C
compiler and Cython are used opportunistically at install time to build the
optional acceleration extension described above; neither is required for
the package to install and work correctly.

## Quick start

```python
import numpy as np
from scaled_vecchia import ScaledVecchiaGP

# X_train: (n, d), y_train: (n,)
gp = ScaledVecchiaGP(m_est=30, m_pred=140, nu=2.5).fit(X_train, y_train)

# Marginal predictive mean and standard deviation
mean, sd = gp.predict(X_test, return_std=True)

# Correlated joint sample paths at a set of new locations
draws = gp.sample_joint(X_path, n_sim=100)   # shape (100, len(X_path))

print(gp.summary())

# To estimate the Matern smoothness instead of fixing it:
gp2 = ScaledVecchiaGP(m_est=30, m_pred=140, nu=None).fit(X_train, y_train)
print(gp2.nu_)   # the fitted smoothness
```

`gp.summary()` prints the fitted variance, smoothness, nugget,
variance-correction factor `b`, and the estimated input *relevances*
`1 / lambda_l` (larger = more influential input), which is a convenient
one-line sensitivity analysis for computer-model emulation.

### Key options on `ScaledVecchiaGP`

| Parameter | Default | Meaning |
|---|---|---|
| `m_est` | 30 | Conditioning-set size used during likelihood optimization. |
| `m_pred` | 140 | Conditioning-set size used at prediction time (larger = more accurate, slower). |
| `n_est` | 5000 | Subsample size used for parameter estimation (fitting is `O(n_est * m_est^3)`). |
| `nu` | 2.5 | Matern smoothness: `0.5`/`1.5`/`2.5` (fast), any other fixed positive float (general Bessel-based), or `None` to estimate it. |
| `trend` | `"constant"` | Mean function: `"zero"`, `"constant"`, or `"linear"`. |
| `nugget` | `None` | `None` estimates a relative nugget; a float (e.g. `1e-8`) fixes it — useful for deterministic computer models. |
| `var_correction` | `True` | Estimate the Sec. 3.4 predictive-variance inflation factor `b` on an inner split. |
| `lambda_max` | `1e3` | Ranges above this are treated as "infinite" (soft variable selection). |
| `n_jobs` | `-1` | OpenMP threads for the optional Cython backend (general/estimated `nu` only); `-1` uses all cores. No effect on the default fixed-`nu` fast path, and no effect at all if the extension wasn't built. |

See the docstring on `ScaledVecchiaGP` for the full list.

### API

- `ScaledVecchiaGP.fit(X, y)` — fit the emulator.
- `ScaledVecchiaGP.predict(X, return_std=False, return_var=False, m=None)` —
  marginal predictive mean (and sd/variance).
- `ScaledVecchiaGP.predict_joint(X, m=None, n_sim=0, exact_var=False)` —
  joint predictive distribution; returns a dict with `mean`, and optionally
  `var` and `samples`.
- `ScaledVecchiaGP.sample_joint(X, n_sim=100, m=None)` — convenience wrapper
  returning just the sample paths, shape `(n_sim, len(X))`.
- `ScaledVecchiaGP.prepare_joint(X, m=None)` — precomputes the joint
  predictive distribution once and returns a `JointPredictiveCache` with a
  cheap `.sample(n_sim, random_state)`, `.mean`, and `.var`. Use this
  instead of repeated `predict_joint`/`sample_joint` calls at a **fixed**
  `X` (e.g. one fresh draw per MCMC/Bayesian-calibration iteration) — see
  "Repeated prediction" below for why, and by how much.
- `ScaledVecchiaGP.summary()` — human-readable fit summary.
- Fitted attributes: `variance_`, `ranges_`, `relevance_`, `nu_`, `nugget_`,
  `beta_`, `loglik_`, `b_`.

Lower-level building blocks are also exported for anyone who wants to
compose their own estimator: `maximin_order`, `find_ordered_nn`,
`vecchia_profile_loglik`.

### Repeated prediction (MCMC / Bayesian-calibration loops)

`predict_joint`/`sample_joint` recompute the whole joint predictive
distribution from scratch on every call — the maximin ordering, every
conditioning set's covariance evaluation and Cholesky factorization, and
the sparse inverse-Cholesky assembly — even though almost none of that
depends on the random draw itself. If you're calling one of them repeatedly
at the **same** test locations (the common pattern when a GP emulator is
queried inside a larger MCMC chain, e.g. via `scaledVecchia4mvBayes`), two
things matter:

1. **Use `prepare_joint` once, then `.sample()` many times.** This skips
   essentially all of the per-call setup on every call after the first —
   measured **~20x faster** for repeated single-sample draws (2000 training
   points, 50 test points, one fresh sample per call: 4.9 ms/call with plain
   `sample_joint` down to 0.24 ms/call with `prepare_joint` + `.sample()`).
   `scaledVecchia4mvBayes`'s wrapper already does this internally, caching
   the setup per distinct `Xtest` — so existing mvBayes-style code gets this
   speedup automatically without any changes.
2. **`random_state=None` (the default) now gives a *fresh* draw every call**,
   not the same one repeated. Previously, an unseeded call reseeded a fresh
   generator from the fixed `self.random_state` (default `0`) every time,
   so every call with default arguments silently returned an *identical*
   sample — a real correctness issue for exactly this repeated-sampling
   pattern, not just a performance one. Draws now come from a persistent
   per-model random stream, seeded once from `random_state` (so the whole
   sequence across a run is still fully reproducible), and advanced on each
   unseeded call. Passing an explicit `random_state` to a specific call
   still gives a fully reproducible, independent draw as before.

Even without switching to `prepare_joint`, plain `predict`/`predict_joint`/
`sample_joint` calls got faster too: the training-data maximin ordering
(`order_obs`), which depends only on the fitted model and not on the test
locations, is now cached after the first joint-prediction call and reused
by every later one (previously recomputed from scratch every call) — this
alone was **~3x** in the same benchmark above. It's invalidated
automatically whenever `fit()` is called again.

## Performance

The two hottest code paths -- the per-block covariance/derivative evaluation
in `_covariance.py` and the maximin ordering in `ordering.py` -- are
implemented with [Numba](https://numba.pydata.org/) (`@njit`) rather than
plain NumPy, since both are dominated by small nested loops that a
vectorised NumPy implementation can't fuse without materialising several
intermediate arrays per block. Measured on the 4000-point/8-D borehole demo
below, this cuts total fit time roughly **3x** compared to an
equivalent pure-NumPy implementation (batched Cholesky/`solve` calls, which
already go through LAPACK, are left as NumPy/SciPy since compiling them
again buys nothing).

Practical notes:

- The first call to a given `@njit` function in a process triggers a JIT
  compile (roughly a few seconds total across all the kernels the package
  uses). `cache=True` persists the compiled code to disk, so this cost is
  paid once per machine/Numba version, not once per run.
- Numba's supported NumPy version range sometimes trails the newest NumPy
  release; if `pip install` reports a resolution conflict, pin NumPy to a
  slightly older minor version or check the
  [Numba compatibility table](https://numba.readthedocs.io/en/stable/user/installing.html).
- `maximin_order` is still the exact `O(n^2 d)` algorithm (just a much
  faster constant factor now); see "Notes / limitations" below.
- For the `nu=None` (or non-half-integer fixed `nu`) path specifically, see
  the optional Cython/OpenMP extension described above and in
  `src/scaled_vecchia/_matern_cy.pyx` -- it's a separate acceleration layer
  from the Numba kernels (Numba doesn't support `scipy.special.kv`), active
  automatically if it was built (`scaled_vecchia.HAVE_CYTHON_BACKEND`).

## Package layout

```
src/scaled_vecchia/
    _covariance.py     # Matern correlation + batched dense linear algebra (Numba)
    _matern_general.py # General (Bessel-based) Matern for non-half-integer/estimated nu
    _matern_cy.pyx      # Optional Cython/OpenMP acceleration for _matern_general.py
    ordering.py         # maximin ordering, ordered nearest-neighbour search
    likelihood.py       # Vecchia loglikelihood, gradient, Fisher information
    optimize.py         # Fisher-scoring optimizer with line search
    gp.py               # ScaledVecchiaGP estimator (fit / predict / predict_joint)
setup.py               # Builds the optional _matern_cy extension, with graceful fallback
tests/                  # pytest test suite
examples/
    borehole_demo.py    # 8-D borehole-function emulation demo (Sec. 4.3 of the paper)
```

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

The test suite checks, among other things:

- the Vecchia log-likelihood reduces exactly to the full-GP log-likelihood
  when the conditioning-set size `m` equals `n - 1` (including with
  `nu=None`);
- the analytic gradient matches finite differences (including the
  finite-differenced `nu` component when estimating it);
- the Fisher information is symmetric and positive (semi-)definite;
- the ordered nearest-neighbour search matches brute force;
- end-to-end `fit`/`predict`/`predict_joint`/`sample_joint` behavior on a
  synthetic smooth function, with both fixed and estimated `nu`;
- if the optional Cython extension was built, that it is numerically
  identical to the pure-Python general-Matern implementation across
  derivatives/`nu`-estimation/nugget-mask combinations, and invariant to
  the number of OpenMP threads used (`tests/test_matern_cy.py`; this file
  is skipped automatically if the extension wasn't built);
- unseeded repeated `sample_joint`/`predict_joint`/`prepare_joint().sample()`
  calls give fresh draws (not a silently repeated one), an explicit
  `random_state` is still fully reproducible, and re-fitting resets the
  random stream reproducibly;
- the training-data ordering cache used by joint prediction doesn't change
  results across different test locations and is correctly invalidated by
  `fit()`;
- `scaledVecchia4mvBayes`'s wrapper caches correctly across repeated calls
  at the same `Xtest`, invalidates when `Xtest` changes, and reproduces the
  same draw sequence given the same `random_state` (`tests/test_mvbayes.py`).

## Demo

```bash
python examples/borehole_demo.py
```

Fits an emulator to the 8-dimensional borehole function (only ~3 inputs
matter), reports RMSE and 95% interval coverage against a held-out test set,
and draws joint sample paths along a path through input space.

## Notes / limitations relative to the paper

- **Ordering.** This implementation uses the simple exact `O(n^2 d)` maximin
  ordering, which is fine up to roughly `n ~ 10^4`–`10^5`. The paper uses the
  quasilinear-time algorithm of Schafer, Sullivan & Owhadi (2021) for larger
  `n`.
- **Covariance.** The fast, Numba-accelerated path covers the isotropic
  Matern family with half-integer smoothness (`nu = 0.5, 1.5, 2.5`, closed
  form, no Bessel calls). Any other fixed `nu`, or `nu=None` to estimate it,
  is supported via a general Bessel-based Matern (`_matern_general.py`),
  optionally accelerated by the Cython/OpenMP extension described above
  (`scaled_vecchia.HAVE_CYTHON_BACKEND`); without it, this path is
  single-threaded and correspondingly slower per block.
- GPU is not supported — everything is vectorized NumPy/SciPy/Numba, plus
  the optional Cython/OpenMP extension for the general-Matern path, all
  batched over conditioning-set blocks.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{katzfuss2020scaled,
  title   = {Scaled {V}ecchia approximation for fast computer-model emulation},
  author  = {Katzfuss, Matthias and Guinness, Joseph and Lawrence, Earl},
  journal = {SIAM/ASA Journal on Uncertainty Quantification},
  volume  = {10},
  number  = {2},
  year    = {2022},
  eprint  = {2005.00386},
  archivePrefix = {arXiv}
}
```

## License

MIT
