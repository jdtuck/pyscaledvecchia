"""Fisher scoring with line-search fallback (Guinness 2021) used to maximise
the Vecchia loglikelihood, Sec. 3.2 of the paper."""

from __future__ import annotations

import numpy as np

__all__ = ["_fisher_scoring"]


def _fisher_scoring(theta0, objective, max_iter=40, tol=1e-4, verbose=False,
                     reorder_hook=None, max_step=3.0):
    """Fisher scoring with line-search fallback (Guinness 2021).

    `objective(theta)` -> (loglik, grad, fisher, beta)
    `reorder_hook(theta, k)` is called at k = 2, 4, 8, 16, ... to refresh the
    maximin ordering / neighbour sets for the current scaled inputs; if it
    returns True the objective has changed and the loglik is recomputed.
    """
    theta = np.array(theta0, dtype=float)
    ll, g, M, beta = objective(theta)
    next_reorder = 2

    for k in range(1, max_iter + 1):
        if reorder_hook is not None and k == next_reorder:
            next_reorder *= 2
            if reorder_hook(theta, k):
                ll, g, M, beta = objective(theta)

        # --- Fisher-scoring direction, ridged until it is an ascent direction
        step = None
        ridge = 0.0
        diag = np.maximum(np.diag(M), 1e-8)
        for _ in range(12):
            try:
                cand = np.linalg.solve(M + ridge * np.diag(diag), g)
            except np.linalg.LinAlgError:
                ridge = max(1e-6, ridge * 10.0)
                continue
            if np.dot(cand, g) > 0 and np.all(np.isfinite(cand)):
                step = cand
                break
            ridge = max(1e-6, ridge * 10.0)
        if step is None:                          # fall back to plain gradient
            step = g / max(np.linalg.norm(g), 1e-12)

        nrm = np.linalg.norm(step)
        if nrm > max_step:
            step *= max_step / nrm

        # --- backtracking line search on the (profiled) loglikelihood -------
        alpha, ll_new, ok = 1.0, ll, False
        for _ in range(20):
            try:
                cand_ll, cand_g, cand_M, cand_beta = objective(theta + alpha * step)
            except np.linalg.LinAlgError:
                alpha *= 0.5
                continue
            if np.isfinite(cand_ll) and cand_ll > ll:
                theta = theta + alpha * step
                ll, g, M, beta = cand_ll, cand_g, cand_M, cand_beta
                ll_new, ok = cand_ll, True
                break
            alpha *= 0.5
        if verbose:
            print(f"    [FS {k:2d}] loglik = {ll:14.4f}   "
                  f"|step.grad| = {abs(np.dot(alpha * step, g)):.3e}")
        if not ok:
            break
        if abs(np.dot(alpha * step, g)) < tol:    # stopping rule from Sec. 3.2
            break

    return theta, ll, beta
