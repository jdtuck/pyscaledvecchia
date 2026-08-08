"""
Isotropic Matern correlation (closed form, half-integer smoothness) and
batched dense linear algebra helpers used to evaluate small conditioning-set
covariance blocks.

References the covariance function of Eq. (1) in Katzfuss, Guinness &
Lawrence (2020/2022), arXiv:2005.00386:

    K(x_i, x_j) = sigma^2 * [ M_nu(q_ij) + tau * 1{i == j} ]
    q_ij        = sqrt( sum_l ((x_il - x_jl) / lambda_l)^2 )
"""

from __future__ import annotations

import math

import numpy as np

LOG2PI = math.log(2.0 * math.pi)
_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)


# ---------------------------------------------------------------------------
# Isotropic Matern correlation for half-integer smoothness
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


def _block_cov(Xb, ranges, variance, nugget, nu, derivs=True, nug_mask=None):
    """Covariance matrices (and d/dlog-parameter derivatives) for a batch of blocks.

    Parameters
    ----------
    Xb        : (B, K, d) raw input coordinates of B blocks of K points.
    ranges    : (d,)  lambda_l
    variance  : sigma^2
    nugget    : tau (relative)
    nug_mask  : (B, K) bool or None. Which diagonal entries receive the nugget.
                None means "all of them". Used for noise-free prediction.

    Returns
    -------
    Sigma  : (B, K, K)
    dSigma : (B, P, K, K) with P = d + 2, derivatives w.r.t.
             log sigma^2, log lambda_1..d, log tau   (or None).
    """
    B, K, d = Xb.shape
    U = Xb / ranges                                   # scaled inputs
    diff = U[:, :, None, :] - U[:, None, :, :]        # (B,K,K,d)
    r = np.sqrt(np.einsum("bijl,bijl->bij", diff, diff))

    M = _matern_corr(r, nu)
    eye = np.eye(K)
    if nug_mask is None:
        nug_diag = eye[None, :, :] * nugget
    else:
        nug_diag = eye[None, :, :] * (nugget * nug_mask[:, :, None])
    Sigma = variance * (M + nug_diag)

    if not derivs:
        return Sigma, None

    P = d + 2
    dS = np.empty((B, P, K, K))
    dS[:, 0] = Sigma                                   # d/d log sigma^2
    g = _matern_g(r, nu)
    for l in range(d):                                 # d/d log lambda_l
        dS[:, 1 + l] = -variance * g * diff[..., l] ** 2
    dS[:, 1 + d] = variance * nug_diag                 # d/d log tau
    return Sigma, dS


# ---------------------------------------------------------------------------
# Batched triangular solves  (numpy has no batched trsm)
# ---------------------------------------------------------------------------

def _forward_solve(L: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    """Solve L X = B for lower-triangular L. L:(b,K,K), B:(b,K,nrhs)."""
    K = L.shape[-1]
    X = np.empty_like(Bm)
    for i in range(K):
        acc = Bm[:, i, :].copy()
        if i:
            acc -= np.einsum("bj,bjr->br", L[:, i, :i], X[:, :i, :])
        X[:, i, :] = acc / L[:, i, i][:, None]
    return X


def _backsolve_LT(L: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    """Solve L^T X = B for lower-triangular L."""
    K = L.shape[-1]
    X = np.empty_like(Bm)
    for i in range(K - 1, -1, -1):
        acc = Bm[:, i, :].copy()
        if i < K - 1:
            acc -= np.einsum("bj,bjr->br", L[:, i + 1:, i], X[:, i + 1:, :])
        X[:, i, :] = acc / L[:, i, i][:, None]
    return X


def _batch_chol(Sigma: np.ndarray) -> np.ndarray:
    """Cholesky with escalating jitter (blocks can be near-singular)."""
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
