# scaled-vecchia

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

The covariance function is an isotropic Matern with half-integer smoothness
(`nu` in `{0.5, 1.5, 2.5}`, closed form — no Bessel function calls) plus a
relative nugget:

```
K(x_i, x_j) = sigma^2 * [ M_nu(q_ij) + tau * 1{i == j} ]
q_ij        = sqrt( sum_l ((x_il - x_jl) / lambda_l)^2 )
```

All parameters (`sigma^2`, `lambda_1..d`, `tau`) are optimized on the log
scale so positivity is automatic.

Pure NumPy/SciPy/Numba — no compiled extension to build (Numba JIT-compiles
the hot loops at runtime and caches them to disk).

## Installation

```bash
pip install -e .
```

or, to also pull in the test dependencies:

```bash
pip install -e ".[test]"
```

Requires Python >= 3.9, NumPy >= 1.22, SciPy >= 1.8, Numba >= 0.58.

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
```

`gp.summary()` prints the fitted variance, nugget, variance-correction
factor `b`, and the estimated input *relevances* `1 / lambda_l` (larger =
more influential input), which is a convenient one-line sensitivity
analysis for computer-model emulation.

### Key options on `ScaledVecchiaGP`

| Parameter | Default | Meaning |
|---|---|---|
| `m_est` | 30 | Conditioning-set size used during likelihood optimization. |
| `m_pred` | 140 | Conditioning-set size used at prediction time (larger = more accurate, slower). |
| `n_est` | 5000 | Subsample size used for parameter estimation (fitting is `O(n_est * m_est^3)`). |
| `nu` | 2.5 | Matern smoothness; one of `0.5`, `1.5`, `2.5`. |
| `trend` | `"constant"` | Mean function: `"zero"`, `"constant"`, or `"linear"`. |
| `nugget` | `None` | `None` estimates a relative nugget; a float (e.g. `1e-8`) fixes it — useful for deterministic computer models. |
| `var_correction` | `True` | Estimate the Sec. 3.4 predictive-variance inflation factor `b` on an inner split. |
| `lambda_max` | `1e3` | Ranges above this are treated as "infinite" (soft variable selection). |

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
- `ScaledVecchiaGP.summary()` — human-readable fit summary.
- Fitted attributes: `variance_`, `ranges_`, `relevance_`, `nugget_`,
  `beta_`, `loglik_`, `b_`.

Lower-level building blocks are also exported for anyone who wants to
compose their own estimator: `maximin_order`, `find_ordered_nn`,
`vecchia_profile_loglik`.

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

## Package layout

```
src/scaled_vecchia/
    _covariance.py   # Matern correlation + batched dense linear algebra
    ordering.py      # maximin ordering, ordered nearest-neighbour search
    likelihood.py     # Vecchia loglikelihood, gradient, Fisher information
    optimize.py       # Fisher-scoring optimizer with line search
    gp.py             # ScaledVecchiaGP estimator (fit / predict / predict_joint)
tests/                # pytest test suite
examples/
    borehole_demo.py  # 8-D borehole-function emulation demo (Sec. 4.3 of the paper)
```

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

The test suite checks, among other things:

- the Vecchia log-likelihood reduces exactly to the full-GP log-likelihood
  when the conditioning-set size `m` equals `n - 1`;
- the analytic gradient matches finite differences;
- the Fisher information is symmetric and positive (semi-)definite;
- the ordered nearest-neighbour search matches brute force;
- end-to-end `fit`/`predict`/`predict_joint`/`sample_joint` behavior on a
  synthetic smooth function.

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
- **Covariance.** Only the isotropic Matern family with half-integer
  smoothness (`nu = 0.5, 1.5, 2.5`) is implemented (closed form, no Bessel
  calls). General `nu` is not supported.
- No compiled/GPU backend — everything is vectorized NumPy/SciPy, batched
  over conditioning-set blocks.

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
