"""
Maximin ordering and ordered nearest neighbours (Sec. 3.1 of the paper).

These are computed *in the scaled input space* x~ = x / lambda so that the
sparsity pattern of the resulting Vecchia approximation adapts to the
estimated anisotropy of the process.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

__all__ = ["maximin_order", "find_ordered_nn"]


def maximin_order(X: np.ndarray, first: int | None = None) -> np.ndarray:
    """Exact maximum-minimum-distance ordering.

    Sequentially picks the point farthest from the set already chosen.
    The first point is the one closest to the centroid.  O(n^2 d); this is
    the simple exact algorithm, adequate up to n ~ 10^4-10^5.  The paper
    uses the quasilinear-time algorithm of Schaefer et al. (2021).
    """
    X = np.ascontiguousarray(X, dtype=float)
    n = X.shape[0]
    if first is None:
        first = int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))
    order = np.empty(n, dtype=np.int64)
    order[0] = first
    d = np.sqrt(((X - X[first]) ** 2).sum(1))
    d[first] = -1.0
    for t in range(1, n):
        j = int(np.argmax(d))
        order[t] = j
        np.minimum(d, np.sqrt(((X - X[j]) ** 2).sum(1)), out=d)
        d[j] = -1.0
    return order


def find_ordered_nn(X: np.ndarray, m: int, start: int = 0,
                     block_size: int = 1024) -> np.ndarray:
    """For each i, the min(i, m) nearest neighbours among indices < i.

    Exact.  Column 0 of the returned array is i itself, columns 1..k the
    neighbours (nearest first is not guaranteed for i > m, order is irrelevant).
    Entries beyond min(i, m) are -1.

    `start` lets us skip rows whose neighbours we do not need (used for the
    observed block during prediction).
    """
    X = np.ascontiguousarray(X, dtype=float)
    n = X.shape[0]
    nn = np.full((n, m + 1), -1, dtype=np.int64)
    nn[:, 0] = np.arange(n)

    lim = min(m + 1, n)                       # rows that condition on *all* previous
    for i in range(max(start, 1), lim):
        prev = np.arange(i)
        dist = np.linalg.norm(X[prev] - X[i], axis=1)
        nn[i, 1:i + 1] = prev[np.argsort(dist)]

    s = max(start, lim)
    while s < n:
        e = min(n, s + block_size)
        b = e - s
        tree = cKDTree(X[:s])
        dd, ii = tree.query(X[s:e], k=min(m, s))
        dd = dd.reshape(b, -1)
        ii = ii.reshape(b, -1)

        # exact distances to the earlier members of this same block
        Dblk = cdist(X[s:e], X[s:e])
        Dblk[np.triu_indices(b)] = np.inf           # only j < i within block
        cand_i = np.concatenate([ii, np.tile(np.arange(s, e), (b, 1))], axis=1)
        cand_d = np.concatenate([dd, Dblk], axis=1)

        take = np.argpartition(cand_d, m - 1, axis=1)[:, :m]
        nn[s:e, 1:m + 1] = np.take_along_axis(cand_i, take, axis=1)
        s = e
    return nn


def _nn_groups(nn: np.ndarray, m: int):
    """Split rows by conditioning-set size so each group can be batched."""
    n = nn.shape[0]
    ks = np.minimum(np.arange(n), m)
    groups = []
    for k in np.unique(ks):
        rows = np.where(ks == k)[0]
        groups.append((rows, nn[rows, 1:k + 1]))
    return groups
