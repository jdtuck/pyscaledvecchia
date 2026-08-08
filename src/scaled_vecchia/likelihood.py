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
"""

from __future__ import annotations

import math

import numpy as np

from ._covariance import LOG2PI, _backsolve_LT, _batch_chol, _block_cov, _chunks, _forward_solve

__all__ = ["vecchia_profile_loglik"]


def _whiten(Xo, Vo, groups, ranges, variance, nugget, nu):
    """Apply the sparse inverse-Cholesky factor U' to the columns of Vo.

    Returns Vt with Vt[i] = a_i' V[U_i], i.e. (V_i - b_i' V_c) / sqrt(d_i),
    plus sum_i log d_i.
    """
    n, q = Vo.shape
    Vt = np.empty((n, q))
    logdet = 0.0
    for rows, nb in groups:
        K = nb.shape[1] + 1
        idx = np.concatenate([nb, rows[:, None]], axis=1)     # (B, K), self last
        for a0, a1 in _chunks(len(rows), K, 1):
            sl = slice(a0, a1)
            Sig, _ = _block_cov(Xo[idx[sl]], ranges, variance, nugget, nu,
                                 derivs=False)
            L = _batch_chol(Sig)
            logdet += 2.0 * np.log(L[:, K - 1, K - 1]).sum()
            E = np.zeros((a1 - a0, K, 1))
            E[:, K - 1, 0] = 1.0
            av = _backsolve_LT(L, E)[:, :, 0]                 # (B, K)
            Vt[rows[sl]] = np.einsum("bk,bkq->bq", av, Vo[idx[sl]])
    return Vt, logdet


def vecchia_profile_loglik(theta, Xo, yo, Zo, groups, nu,
                            need_grad=True, var_penalty=0.0, log_var_target=0.0):
    """Profile Vecchia loglikelihood, its gradient and the Fisher information.

    theta : (d+2,) = log(sigma^2), log(lambda_1..d), log(tau)
    Xo    : (n, d) inputs **in Vecchia order**
    yo    : (n,)   responses in Vecchia order
    Zo    : (n, q) mean-function design matrix in Vecchia order, or None
    groups: output of ordering._nn_groups
    """
    theta = np.asarray(theta, dtype=float)
    d = Xo.shape[1]
    P = d + 2
    variance = math.exp(theta[0])
    ranges = np.exp(theta[1:1 + d])
    nugget = math.exp(theta[1 + d])
    n = Xo.shape[0]

    # ---- pass 1: GLS for beta (profiling), cheap (no derivatives) ----------
    if Zo is not None and Zo.shape[1] > 0:
        Vt, _ = _whiten(Xo, np.column_stack([yo, Zo]), groups,
                         ranges, variance, nugget, nu)
        yt, Zt = Vt[:, 0], Vt[:, 1:]
        XtX = Zt.T @ Zt
        beta = np.linalg.solve(XtX + 1e-12 * np.eye(XtX.shape[0]), Zt.T @ yt)
        resid = yo - Zo @ beta
    else:
        beta = np.zeros(0)
        resid = yo

    # ---- pass 2: loglik / gradient / Fisher information -------------------
    ll = -0.5 * n * LOG2PI
    grad = np.zeros(P)
    fish = np.zeros((P, P))

    for rows, nb in groups:
        K = nb.shape[1] + 1
        idx = np.concatenate([nb, rows[:, None]], axis=1)
        for a0, a1 in _chunks(len(rows), K, P if need_grad else 1):
            sl = slice(a0, a1)
            block = idx[sl]
            Sig, dS = _block_cov(Xo[block], ranges, variance, nugget, nu,
                                  derivs=need_grad)
            L = _batch_chol(Sig)
            z = _forward_solve(L, resid[block][:, :, None])[:, :, 0]   # (B,K)
            zK = z[:, K - 1]
            ll -= 0.5 * (2.0 * np.log(L[:, K - 1, K - 1]) + zK ** 2).sum()
            if not need_grad:
                continue
            E = np.zeros((a1 - a0, K, 1))
            E[:, K - 1, 0] = 1.0
            av = _backsolve_LT(L, E)[:, :, 0]                          # (B,K)
            rhs = np.einsum("bpkl,bl->bkp", dS, av)                    # (B,K,P)
            W = _forward_solve(L, rhs).transpose(0, 2, 1)              # (B,P,K)
            wK = W[:, :, K - 1]                                        # (B,P)
            wz = np.einsum("bpk,bk->bp", W, z)                         # (B,P)
            grad -= 0.5 * (wK * (1.0 + zK[:, None] ** 2)
                           - 2.0 * zK[:, None] * wz).sum(0)
            fish += 0.5 * (2.0 * np.einsum("bpk,bqk->pq", W, W)
                           - wK.T @ wK)

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
