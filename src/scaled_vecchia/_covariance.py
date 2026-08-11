"""
Isotropic Matern correlation (closed form, half-integer smoothness) and
batched dense linear algebra helpers used to evaluate small conditioning-set
covariance blocks.

References the covariance function of Eq. (1) in Katzfuss, Guinness &
Lawrence (2020/2022), arXiv:2005.00386:

    K(x_i, x_j) = sigma^2 * [ M_nu(q_ij) + tau * 1{i == j} ]
    q_ij        = sqrt( sum_l ((x_il - x_jl) / lambda_l)^2 )

Performance
-----------
`_block_cov` and the batched triangular solves below are, by a wide margin,
the hottest code in the package (see the package README for profiling
details): they are called once per conditioning-set block on every
likelihood/gradient evaluation. They are implemented with Numba (`@njit`)
so that the whole computation for each block is a single fused native loop
with no intermediate array allocation, rather than the half-dozen separate
NumPy passes (difference tensor, distance, Matern correlation, per-dimension
derivative) a pure-NumPy implementation needs. Measured speedups vs. the
equivalent vectorised NumPy code: ~3x for `_block_cov`, ~2x for the
triangular solves, and ~7-11x for the maximin ordering in `ordering.py`.

`_matern_corr` / `_matern_g` are kept as plain NumPy reference
implementations (used directly by the test suite's brute-force
exact-likelihood check) and are not on the hot path themselves.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._matern_general import _block_cov_general

LOG2PI = math.log(2.0 * math.pi)
_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)


# ---------------------------------------------------------------------------
# Isotropic Matern correlation for half-integer smoothness (reference /
# non-hot-path NumPy implementations; also used directly by the tests)
# ---------------------------------------------------------------------------

def _matern_corr(r: np.ndarray, nu: float) -> np.ndarray:
    """M_nu(r): isotropic Matern correlation, closed form for nu = .5/1.5/2.5."""
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        c = _SQRT3 * r
        return (1.0 + c) * np.exp(-c)
    if nu == 2.5:
        c = _SQRT5 * r
        return (1.0 + c + c * c / 3.0) * np.exp(-c)
    raise ValueError("nu must be one of 0.5, 1.5, 2.5")


def _matern_g(r: np.ndarray, nu: float) -> np.ndarray:
    """g(r) = M_nu'(r) / r  (finite at r=0 for nu >= 1.5)."""
    if nu == 0.5:
        rs = np.where(r > 0.0, r, 1.0)
        return np.where(r > 0.0, -np.exp(-rs) / rs, 0.0)
    if nu == 1.5:
        return -3.0 * np.exp(-_SQRT3 * r)
    if nu == 2.5:
        c = _SQRT5 * r
        return -(5.0 / 3.0) * (1.0 + c) * np.exp(-c)
    raise ValueError("nu must be one of 0.5, 1.5, 2.5")


def _check_nu(nu: float) -> None:
    if not (isinstance(nu, (int, float)) and nu > 0.0):
        raise ValueError("nu must be a positive number")


# ---------------------------------------------------------------------------
# Fused Numba kernels: covariance (+ derivatives) for a batch of blocks.
# Each (b, i, j) entry is computed in a single native loop -- no (B,K,K,d)
# difference tensor, no separate Matern/derivative arrays.
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def _block_sigma_kernel(Xb, ranges, variance, nugget, nu, nug_mask):
    B, K, D = Xb.shape
    Sigma = np.empty((B, K, K))
    for b in prange(B):
        for i in range(K):
            for j in range(K):
                s = 0.0
                for l in range(D):
                    diff = (Xb[b, i, l] - Xb[b, j, l]) / ranges[l]
                    s += diff * diff
                r = math.sqrt(s)
                if nu == 0.5:
                    M = math.exp(-r)
                elif nu == 1.5:
                    c = _SQRT3 * r
                    M = (1.0 + c) * math.exp(-c)
                else:
                    c = _SQRT5 * r
                    M = (1.0 + c + c * c / 3.0) * math.exp(-c)
                nug = nugget * nug_mask[b, i] if i == j else 0.0
                Sigma[b, i, j] = variance * (M + nug)
    return Sigma


@njit(cache=True, fastmath=True, parallel=True)
def _block_cov_grad_kernel(Xb, ranges, variance, nugget, nu, nug_mask):
    B, K, D = Xb.shape
    P = D + 2
    Sigma = np.empty((B, K, K))
    dS = np.empty((B, P, K, K))
    for b in prange(B):
        for i in range(K):
            for j in range(K):
                s = 0.0
                for l in range(D):
                    diff = (Xb[b, i, l] - Xb[b, j, l]) / ranges[l]
                    s += diff * diff
                r = math.sqrt(s)
                if nu == 0.5:
                    if r > 0.0:
                        M = math.exp(-r)
                        g = -math.exp(-r) / r
                    else:
                        M = 1.0
                        g = 0.0
                elif nu == 1.5:
                    c = _SQRT3 * r
                    M = (1.0 + c) * math.exp(-c)
                    g = -3.0 * math.exp(-c)
                else:
                    c = _SQRT5 * r
                    M = (1.0 + c + c * c / 3.0) * math.exp(-c)
                    g = -(5.0 / 3.0) * (1.0 + c) * math.exp(-c)
                nug = nugget * nug_mask[b, i] if i == j else 0.0
                Sigma[b, i, j] = variance * (M + nug)
                dS[b, 0, i, j] = Sigma[b, i, j]
                for l in range(D):
                    diffl = (Xb[b, i, l] - Xb[b, j, l]) / ranges[l]
                    dS[b, 1 + l, i, j] = -variance * g * diffl * diffl
                dS[b, 1 + D, i, j] = variance * nug
    return Sigma, dS


def _block_cov(Xb, ranges, variance, nugget, nu, derivs=True, nug_mask=None):
    """Covariance matrices (and d/dlog-parameter derivatives) for a batch of blocks.

    Parameters
    ----------
    Xb        : (B, K, d) raw input coordinates of B blocks of K points.
    ranges    : (d,)  lambda_l
    variance  : sigma^2
    nugget    : tau (relative)
    nu        : Matern smoothness. nu in {0.5, 1.5, 2.5} uses the fast,
                Numba-compiled closed-form kernels; any other positive value
                transparently falls back to the general Bessel-based Matern
                (`_matern_general._block_cov_general`), which is not
                Numba-accelerated and is correspondingly slower per block.
                To *estimate* nu rather than fix it, see `ScaledVecchiaGP`
                (which handles the extra derivative w.r.t. nu itself via
                `_block_cov_general(..., estimate_nu=True)` directly).
    nug_mask  : (B, K) bool/float or None. Which diagonal entries receive the
                nugget. None means "all of them". Used for noise-free
                prediction.

    Returns
    -------
    Sigma  : (B, K, K)
    dSigma : (B, P, K, K) with P = d + 2, derivatives w.r.t.
             log sigma^2, log lambda_1..d, log tau   (or None).
    """
    _check_nu(nu)
    if nu not in (0.5, 1.5, 2.5):
        # General smoothness fixed (not estimated): use the Bessel-based
        # covariance, but without the extra nu-derivative row.
        return _block_cov_general(Xb, ranges, variance, nugget, nu,
                                   derivs=derivs, nug_mask=nug_mask,
                                   estimate_nu=False)

    B, K, _d = Xb.shape
    Xb = np.ascontiguousarray(Xb, dtype=np.float64)
    ranges = np.ascontiguousarray(ranges, dtype=np.float64)
    mask = (np.ones((B, K)) if nug_mask is None
            else np.ascontiguousarray(nug_mask, dtype=np.float64))

    if not derivs:
        Sigma = _block_sigma_kernel(Xb, ranges, float(variance), float(nugget),
                                     float(nu), mask)
        return Sigma, None

    Sigma, dS = _block_cov_grad_kernel(Xb, ranges, float(variance), float(nugget),
                                        float(nu), mask)
    return Sigma, dS


# ---------------------------------------------------------------------------
# Batched triangular solves  (NumPy has no batched trsm)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=True)
def _forward_solve(L: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    """Solve L X = B for lower-triangular L. L:(b,K,K), B:(b,K,nrhs)."""
    b, K, nrhs = Bm.shape
    X = np.empty_like(Bm)
    for bi in prange(b):
        for r in range(nrhs):
            for i in range(K):
                acc = Bm[bi, i, r]
                for j in range(i):
                    acc -= L[bi, i, j] * X[bi, j, r]
                X[bi, i, r] = acc / L[bi, i, i]
    return X


@njit(cache=True, fastmath=True, parallel=True)
def _backsolve_LT(L: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    """Solve L^T X = B for lower-triangular L."""
    b, K, nrhs = Bm.shape
    X = np.empty_like(Bm)
    for bi in prange(b):
        for r in range(nrhs):
            for i in range(K - 1, -1, -1):
                acc = Bm[bi, i, r]
                for j in range(i + 1, K):
                    acc -= L[bi, j, i] * X[bi, j, r]
                X[bi, i, r] = acc / L[bi, i, i]
    return X


def _batch_chol(Sigma: np.ndarray) -> np.ndarray:
    """Cholesky with escalating jitter (blocks can be near-singular).

    Delegates to `np.linalg.cholesky` (LAPACK): already compiled, batched,
    and not a significant fraction of runtime, so left as NumPy.
    """
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        pass
    scale = np.einsum("bii->b", Sigma) / Sigma.shape[-1]
    eye = np.eye(Sigma.shape[-1])
    for p in range(-12, -3):
        try:
            return np.linalg.cholesky(Sigma + (10.0 ** p) * scale[:, None, None] * eye)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("block Cholesky failed even with jitter")


def _chunks(n_rows: int, K: int, P: int, budget: int = 4_000_000):
    """Yield (start, end) row-slices sized so that K*K*P*chunk stays bounded."""
    step = max(1, budget // max(1, K * K * max(P, 1)))
    for a in range(0, n_rows, step):
        yield a, min(n_rows, a + step)
