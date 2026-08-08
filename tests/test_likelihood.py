"""
Correctness tests for the Vecchia loglikelihood:

(a) with m = n-1 the Vecchia likelihood must equal the exact GP likelihood.
(b) the analytic gradient must match finite differences.
(c) the Fisher information must be symmetric.
"""

import math

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from scaled_vecchia._covariance import LOG2PI, _matern_corr
from scaled_vecchia.likelihood import vecchia_profile_loglik
from scaled_vecchia.ordering import _nn_groups, find_ordered_nn, maximin_order


def _exact_loglik(X, y, theta, nu):
    d = X.shape[1]
    var, ranges, nug = math.exp(theta[0]), np.exp(theta[1:1 + d]), math.exp(theta[-1])
    Xs = X / ranges
    r = cdist(Xs, Xs)
    Kmat = var * (_matern_corr(r, nu) + nug * np.eye(len(X)))
    L = np.linalg.cholesky(Kmat)
    a = np.linalg.solve(L, y)
    return -0.5 * (len(y) * LOG2PI + 2 * np.log(np.diag(L)).sum() + a @ a)


@pytest.fixture
def toy_problem():
    rng = np.random.default_rng(1)
    n, d, nu = 60, 3, 2.5
    X = rng.random((n, d))
    theta = np.array([math.log(1.7), math.log(0.35), math.log(0.9),
                       math.log(3.0), math.log(0.05)])
    y = rng.standard_normal(n)
    return X, y, theta, nu


def test_vecchia_matches_exact_when_m_equals_n_minus_1(toy_problem):
    X, y, theta, nu = toy_problem
    n, d = X.shape
    m = n - 1
    order = maximin_order(X / np.exp(theta[1:1 + d]))
    nn = find_ordered_nn((X / np.exp(theta[1:1 + d]))[order], m)
    groups = _nn_groups(nn, m)
    ll_v, _, _, _ = vecchia_profile_loglik(theta, X[order], y[order], None,
                                            groups, nu)
    ll_e = _exact_loglik(X, y, theta, nu)
    assert abs(ll_v - ll_e) < 1e-7


def test_gradient_matches_finite_differences(toy_problem):
    X, y, theta, nu = toy_problem
    n, d = X.shape
    m = 10
    order = maximin_order(X / np.exp(theta[1:1 + d]))
    nn = find_ordered_nn((X / np.exp(theta[1:1 + d]))[order], m)
    groups = _nn_groups(nn, m)
    Z = np.ones((n, 1))

    def f(th):
        return vecchia_profile_loglik(th, X[order], y[order], Z[order],
                                       groups, nu)

    ll0, g0, M0, beta0 = f(theta)
    num = np.empty_like(g0)
    h = 1e-6
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        num[j] = (f(tp)[0] - f(tm)[0]) / (2 * h)
    err = np.max(np.abs(g0 - num) / (1 + np.abs(num)))
    assert err < 1e-5

    # Fisher information should be (numerically) symmetric.
    assert np.abs(M0 - M0.T).max() < 1e-6
    # ... and positive (semi-)definite.
    ev = np.linalg.eigvalsh(0.5 * (M0 + M0.T))
    assert ev.min() > -1e-6


def test_loglik_without_grad_matches_with_grad(toy_problem):
    X, y, theta, nu = toy_problem
    n, d = X.shape
    m = 10
    order = maximin_order(X / np.exp(theta[1:1 + d]))
    nn = find_ordered_nn((X / np.exp(theta[1:1 + d]))[order], m)
    groups = _nn_groups(nn, m)

    ll_grad, _, _, _ = vecchia_profile_loglik(theta, X[order], y[order], None,
                                               groups, nu, need_grad=True)
    ll_nograd, g, M, _ = vecchia_profile_loglik(theta, X[order], y[order], None,
                                                 groups, nu, need_grad=False)
    assert g is None and M is None
    assert abs(ll_grad - ll_nograd) < 1e-10
