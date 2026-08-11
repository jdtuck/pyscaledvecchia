"""
Vecchia loglikelihood, gradient and Fisher information (Sec. 3.2 of the
paper), plus the whitening transform used both to profile out the linear
mean-function coefficients by GLS and to estimate the predictive-variance
correction of Sec. 3.4.

Following Eq. (4) of the paper we write the Vecchia loglikelihood as a sum of
differences of two Gaussian log-densities.  For block U_i = (c(i), i) with
covariance S = K(U_i, U_i) and Cholesky S = L L',

    z = L^{-1} r_{U_i},   a = L^{-T} e_K

Because the Cholesky of the leading block of S is the leading block of L, all
the "minus the conditioning-set density" terms telescope into the last
row/column only:

    logdet S - logdet S_c        = 2 log L_KK
    r' S^-1 r - r_c' S_c^-1 r_c  = z_K^2

so   l_i = -1/2 ( log 2pi + 2 log L_KK + z_K^2 ).

With A^j = L^{-1} (dS/dtheta_j) L^{-T} and w^j = row K of A^j = L^{-1}(dS_j a):

    dl_i/dtheta_j = -1/2 [ w^j_K (1 + z_K^2) - 2 z_K (w^j . z) ]
    I_i[j,k]      =  1/2 [ 2 (w^j . w^k) - w^j_K w^k_K ]

Only the K-th row of A^j is ever needed, so each parameter costs O(K^2).
Bonus: a = ( -b_i / sqrt(d_i) , 1 / sqrt(d_i) ) is exactly the i-th column of
the sparse inverse-Cholesky factor U, which we reuse for prediction.

Performance note
-----------------
A naive two-pass implementation would (1) whiten [y, Z] with a
no-derivatives covariance evaluation to profile out beta by GLS, then
(2) re-evaluate the *same* conditioning-set covariance blocks *with*
derivatives to get the gradient/Fisher information -- i.e. the (expensive)
Matern evaluation and batched Cholesky factorisation of every block would
run twice per Fisher-scoring iteration.

Instead we exploit linearity of the whitening map L^{-1}(.): for a fixed
block Cholesky factor L,

    z = L^{-1}(y - Z beta) = L^{-1} y - (L^{-1} Z) beta = yt - Zt @ beta

so once we have cached the *fully whitened* yt/Zt columns (and, for the
gradient, the small (P, K) matrix W = L^{-1}(dS L^{-T} e_K)) we can recover
the whitened residual z for *any* candidate beta with plain array algebra --
no second covariance evaluation or Cholesky factorisation required.  The
Fisher information does not depend on beta or the residuals at all, so it is
accumulated directly in the first pass.  Net effect: one Matern evaluation
and one batched Cholesky per block per objective call, instead of two.
"""

from __future__ import annotations

import math

import numpy as np

from ._covariance import LOG2PI, _backsolve_LT, _batch_chol, _block_cov, _chunks, _forward_solve
from ._matern_general import _block_cov_general

__all__ = ["vecchia_profile_loglik"]


def vecchia_profile_loglik(theta, Xo, yo, Zo, groups, nu, estimate_nu=False,
                            need_grad=True, var_penalty=0.0, log_var_target=0.0):
    """Profile Vecchia loglikelihood, its gradient and the Fisher information.

    theta : (P,) parameter vector on the log scale. Layout is
            [log sigma^2, log lambda_1..d, log tau] (P = d+2) when
            `estimate_nu` is False (`nu` is then the *fixed* smoothness), or
            [log sigma^2, log lambda_1..d, log nu, log tau] (P = d+3) when
            `estimate_nu` is True (`nu` is then only the current iterate's
            value on the *natural* scale, used only for reference/logging by
            the caller -- theta[1+d] is authoritative).
    Xo    : (n, d) inputs **in Vecchia order**
    yo    : (n,)   responses in Vecchia order
    Zo    : (n, q) mean-function design matrix in Vecchia order, or None
    groups: output of ordering._nn_groups
    estimate_nu : if True, use the general Bessel-based Matern covariance and
            estimate nu itself as an extra parameter (Sec. 3.5 of the paper;
            see `_matern_general.py`). Not Numba-accelerated, so noticeably
            slower per block than the default fixed nu in {0.5, 1.5, 2.5}.
    """
    theta = np.asarray(theta, dtype=float)
    d = Xo.shape[1]
    variance = math.exp(theta[0])
    ranges = np.exp(theta[1:1 + d])
    if estimate_nu:
        nu = math.exp(theta[1 + d])
        nugget = math.exp(theta[2 + d])
        P = d + 3
    else:
        nugget = math.exp(theta[1 + d])
        P = d + 2
    n = Xo.shape[0]

    has_Z = Zo is not None and Zo.shape[1] > 0
    q = Zo.shape[1] if has_Z else 0

    XtX = np.zeros((q, q))
    Zty = np.zeros(q)
    fish = np.zeros((P, P))

    # ---- single pass: evaluate each block's covariance (+derivatives) once,
    # accumulate the GLS normal equations and (beta-independent) Fisher
    # information, and cache the small per-block quantities needed to finish
    # the loglikelihood/gradient once beta is known. ------------------------
    cache = []
    for rows, nb in groups:
        K = nb.shape[1] + 1
        idx = np.concatenate([nb, rows[:, None]], axis=1)     # (B, K), self last
        for a0, a1 in _chunks(len(rows), K, P if need_grad else 1):
            sl = slice(a0, a1)
            block = idx[sl]
            if estimate_nu:
                Sig, dS = _block_cov_general(Xo[block], ranges, variance, nugget,
                                              nu, derivs=need_grad, estimate_nu=True)
            else:
                Sig, dS = _block_cov(Xo[block], ranges, variance, nugget, nu,
                                      derivs=need_grad)
            L = _batch_chol(Sig)

            E = np.zeros((a1 - a0, K, 1))
            E[:, K - 1, 0] = 1.0
            av = _backsolve_LT(L, E)[:, :, 0]                 # (B, K)

            if has_Z:
                V = np.concatenate([yo[block][:, :, None], Zo[block]], axis=2)
                Vt = np.einsum("bk,bkq->bq", av, V)           # K-th component only
                yt, Zt = Vt[:, 0], Vt[:, 1:]
                XtX += Zt.T @ Zt
                Zty += Zt.T @ yt

            W = None
            if need_grad:
                rhs = np.einsum("bpkl,bl->bkp", dS, av)                # (B,K,P)
                W = _forward_solve(L, rhs).transpose(0, 2, 1)          # (B,P,K)
                wK = W[:, :, K - 1]                                    # (B,P)
                fish += 0.5 * (2.0 * np.einsum("bpk,bqk->pq", W, W)
                               - wK.T @ wK)

            cache.append((block, K, L, W))

    beta = np.linalg.solve(XtX + 1e-12 * np.eye(q), Zty) if has_Z else np.zeros(0)

    # ---- finish loglikelihood / gradient using the cached factors ---------
    ll = -0.5 * n * LOG2PI
    grad = np.zeros(P)

    for block, K, L, W in cache:
        resid_block = (yo[block] - Zo[block] @ beta) if has_Z else yo[block]
        z = _forward_solve(L, resid_block[:, :, None])[:, :, 0]        # (B,K)
        zK = z[:, K - 1]
        ll -= 0.5 * (2.0 * np.log(L[:, K - 1, K - 1]) + zK ** 2).sum()
        if not need_grad:
            continue
        wK = W[:, :, K - 1]                                            # (B,P)
        wz = np.einsum("bpk,bk->bp", W, z)                             # (B,P)
        grad -= 0.5 * (wK * (1.0 + zK[:, None] ** 2)
                       - 2.0 * zK[:, None] * wz).sum(0)

    # ---- mild penalty discouraging sigma^2 >> sample variance (Sec. 3.2) ---
    if var_penalty > 0.0:
        excess = max(0.0, theta[0] - log_var_target)
        ll -= 0.5 * var_penalty * excess ** 2
        if need_grad:
            grad[0] -= var_penalty * excess
            if excess > 0.0:
                fish[0, 0] += var_penalty

    if not need_grad:
        return ll, None, None, beta
    return ll, grad, fish, beta
