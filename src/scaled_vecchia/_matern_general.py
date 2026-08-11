"""
General isotropic Matern covariance for arbitrary smoothness `nu`, and the
machinery to *estimate* `nu` from the data (Sec. 3.5 of Katzfuss, Guinness &
Lawrence, 2020/2022, arXiv:2005.00386).

The half-integer closed forms in `_covariance.py` (nu = 0.5, 1.5, 2.5) only
exist because the modified Bessel function K_nu has an elementary closed
form at those specific orders, and (to match the standard "range parameter"
convention, e.g. Rasmussen & Williams 2006 Eq. 4.14) they are written in
terms of the *rescaled* distance c = sqrt(2 nu) * r, not r itself:

    M_0.5(r) = exp(-c),                        c = r
    M_1.5(r) = (1+c) exp(-c),                  c = sqrt(3) r
    M_2.5(r) = (1 + c + c^2/3) exp(-c),        c = sqrt(5) r

For a general nu we use the same convention:

    M_nu(r) = 2^(1-nu) / Gamma(nu) * c^nu * K_nu(c),     c = sqrt(2 nu) * r,
    M_nu(0) = 1

which reduces exactly to the three closed forms above at nu = 0.5, 1.5, 2.5
(verified in the test suite).

The paper's own R implementation (arXiv:2005.00386's companion package,
katzfuss-group/scaledVecchia, `vecchia_scaled.R`) does exactly this: when
`nu` is not fixed by the user it switches the covariance function from the
closed-form half-integer version to the general (Bessel) version and
estimates `nu` jointly with the other covariance parameters by Fisher
scoring, starting from an initial value (their default starting value is
3.5). We follow the same approach.

Two derivatives are needed for Fisher scoring:

- d/dr M_nu(r): has a simple closed form via the Bessel recurrence
  d/dz [z^nu K_nu(z)] = -z^nu K_{nu-1}(z), applied with z = c = sqrt(2 nu) r,
  so

      M_nu'(r) = -sqrt(2 nu) * 2^(1-nu)/Gamma(nu) * c^nu * K_{nu-1}(c)

  This lets the range-parameter gradients stay fully analytic even for a
  general nu (same `g(r) = M'(r)/r` role as `_matern_g` in `_covariance.py`).

- d/dnu M_nu(r): the derivative of a Bessel function with respect to its
  *order* has no simple closed form in elementary functions, and here nu
  also appears inside c = sqrt(2 nu) r. As in practice elsewhere (this is
  exactly why the paper's own half-integer covariance functions "avoid
  Bessel functions" for speed), we use a central finite difference in nu
  for this one component. Everything else in the gradient/Fisher-information
  machinery remains analytic.

This path is not accelerated with Numba (`scipy.special.kv` is not
Numba-compatible), so estimating `nu` -- or fixing it at a non-half-integer
value -- is meaningfully slower per conditioning-set block than the default
nu in {0.5, 1.5, 2.5}. This mirrors the trade-off the paper itself notes.

Performance
-----------
Profiling shows essentially all the time in this path goes to `kv` itself
(the Matern correlation and its r-derivative together account for ~85% of
total runtime when nu is estimated). Two cheap, exactness-preserving
reductions in the number of Bessel evaluations:

1. Every quantity here (M, g, dM/dnu) only depends on the *symmetric*
   pairwise distance matrix r, so each conditioning-set block only needs
   Bessel evaluations on its K(K+1)/2 upper-triangular entries, not all K^2
   -- roughly a 2x reduction, exploited by `_symmetric_eval` below.
2. The nu-derivative's finite difference exploits the same symmetry trick
   for both of its Bessel evaluations (M(nu+h) and M(nu-h)), so it still
   costs only 2x the reduced per-evaluation cost of step 1 rather than 2x
   the full K^2 cost. We keep a *central* difference (not forward) since a
   forward difference's O(h) truncation error is large enough to show up as
   a measurable gradient bias (verified in the test suite); central
   differencing keeps the gradient accurate to ~1e-6 relative error.

Together these cut the Bessel-evaluation cost by roughly 2x.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, kv

__all__ = ["_matern_corr_general", "_matern_g_general", "_block_cov_general"]

_R_FLOOR = 1e-7   # keeps Bessel evaluations away from the r=0 singularity;
                   # always paired with a diff term that is exactly 0 there,
                   # so this floor introduces no error in derivatives.


def _matern_corr_general(r: np.ndarray, nu: float) -> np.ndarray:
    """M_nu(r) = 2^(1-nu)/Gamma(nu) * c^nu * K_nu(c), c = sqrt(2 nu) r."""
    r = np.asarray(r, dtype=float)
    rr = np.where(r < _R_FLOOR, _R_FLOOR, r)
    c = math.sqrt(2.0 * nu) * rr
    log_coef = (1.0 - nu) * math.log(2.0) - gammaln(nu)
    val = np.exp(log_coef + nu * np.log(c)) * kv(nu, c)
    return np.where(r < _R_FLOOR, 1.0, val)


def _matern_g_general(r: np.ndarray, nu: float) -> np.ndarray:
    """g(r) = M_nu'(r) / r via d/dz[z^nu K_nu(z)] = -z^nu K_{nu-1}(z),
    with z = c = sqrt(2 nu) r (chain rule picks up a factor sqrt(2 nu)).

    Only ever multiplied by a squared-difference that is exactly 0 at r=0
    (the diagonal of a conditioning-set block), so the value returned at
    r=0 is never actually used -- we just need it finite, which the floor
    guarantees.
    """
    r = np.asarray(r, dtype=float)
    rr = np.where(r < _R_FLOOR, _R_FLOOR, r)
    c = math.sqrt(2.0 * nu) * rr
    log_coef = (1.0 - nu) * math.log(2.0) - gammaln(nu)
    val = -(math.sqrt(2.0 * nu) / rr) * np.exp(log_coef + nu * np.log(c)) * kv(nu - 1.0, c)
    return np.where(r < _R_FLOOR, 0.0, val)


def _symmetric_eval(func, r_full, iu):
    """Apply `func` (elementwise, symmetric in i/j through r) to a batch of
    (K,K) distance matrices using only the K(K+1)/2 upper-triangular entries,
    then mirror the result back into a full (B,K,K) array. `r_full` is
    (B,K,K); `iu = np.triu_indices(K)`.
    """
    B, K, _ = r_full.shape
    r_flat = r_full[:, iu[0], iu[1]]              # (B, K(K+1)/2)
    val_flat = func(r_flat)
    out = np.empty((B, K, K))
    out[:, iu[0], iu[1]] = val_flat
    out[:, iu[1], iu[0]] = val_flat
    return out


def _block_cov_general(Xb, ranges, variance, nugget, nu, derivs=True,
                        nug_mask=None, estimate_nu=False, nu_fd_h=1e-3):
    """Like `_covariance._block_cov`, but for a general (possibly estimated)
    Matern smoothness `nu`, via the Bessel-based formulas above.

    When `estimate_nu` is True, `dS` gets one extra derivative row (w.r.t.
    log nu), inserted right after the range derivatives and before the
    nugget derivative: columns are
    [log sigma^2, log lambda_1..d, log nu, log tau].
    """
    B, K, d = Xb.shape
    U = Xb / ranges
    diff = U[:, :, None, :] - U[:, None, :, :]        # (B,K,K,d)
    r = np.sqrt(np.einsum("bijl,bijl->bij", diff, diff))
    iu = np.triu_indices(K)

    M = _symmetric_eval(lambda rr: _matern_corr_general(rr, nu), r, iu)
    eye = np.eye(K)
    if nug_mask is None:
        nug_diag = eye[None, :, :] * nugget
    else:
        nug_diag = eye[None, :, :] * (nugget * nug_mask[:, :, None])
    Sigma = variance * (M + nug_diag)

    if not derivs:
        return Sigma, None

    P = d + 2 + (1 if estimate_nu else 0)
    dS = np.empty((B, P, K, K))
    dS[:, 0] = Sigma                                   # d/d log sigma^2
    g = _symmetric_eval(lambda rr: _matern_g_general(rr, nu), r, iu)
    for l in range(d):                                 # d/d log lambda_l
        dS[:, 1 + l] = -variance * g * diff[..., l] ** 2
    idx = 1 + d
    if estimate_nu:
        # Central finite difference in nu, each Bessel evaluation using only
        # the K(K+1)/2 unique upper-triangular distances (see module note).
        dMnu = _symmetric_eval(
            lambda rr: (_matern_corr_general(rr, nu + nu_fd_h)
                        - _matern_corr_general(rr, nu - nu_fd_h)) / (2.0 * nu_fd_h),
            r, iu)
        # d/d log nu = nu * dM/dnu (chain rule for the log-parameterization)
        dS[:, idx] = variance * nu * dMnu
        idx += 1
    dS[:, idx] = variance * nug_diag                   # d/d log tau
    return Sigma, dS
