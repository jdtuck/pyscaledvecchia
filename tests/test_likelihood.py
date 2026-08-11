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


# ---------------------------------------------------------------------------
# General (Bessel-based) Matern / nu estimation
# ---------------------------------------------------------------------------

def test_general_matern_reduces_to_closed_form_at_half_integers():
    from scaled_vecchia._matern_general import _matern_corr_general, _matern_g_general
    r = np.linspace(1e-6, 5, 200)
    for nu in (0.5, 1.5, 2.5):
        np.testing.assert_allclose(_matern_corr_general(r, nu), _matern_corr(r, nu),
                                    atol=1e-10)
        # g(r) has a removable singularity handled differently very close to
        # r=0 in the two implementations; compare away from it.
        mask = r > 0.01
        from scaled_vecchia._covariance import _matern_g
        np.testing.assert_allclose(_matern_g_general(r, nu)[mask], _matern_g(r, nu)[mask],
                                    atol=1e-8)


def test_general_matern_g_matches_numerical_derivative_for_noninteger_nu():
    from scaled_vecchia._matern_general import _matern_corr_general, _matern_g_general
    r = np.linspace(0.05, 5, 200)
    nu = 1.75  # no closed form -- genuinely exercises the general Bessel path
    h = 1e-6
    g_num = (_matern_corr_general(r + h, nu) - _matern_corr_general(r - h, nu)) / (2 * h) / r
    g_ana = _matern_g_general(r, nu)
    rel_err = np.abs(g_num - g_ana) / np.abs(g_num)
    assert rel_err.max() < 1e-6


def test_estimate_nu_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    n, d, m = 60, 3, 10
    X = rng.random((n, d))
    y = rng.standard_normal(n)
    Z = np.ones((n, 1))
    theta = np.array([math.log(1.7), math.log(0.35), math.log(0.9),
                       math.log(0.5), math.log(1.8), math.log(0.05)])

    order = maximin_order(X / np.exp(theta[1:1 + d]))
    nn = find_ordered_nn((X / np.exp(theta[1:1 + d]))[order], m)
    groups = _nn_groups(nn, m)

    def f(th):
        return vecchia_profile_loglik(th, X[order], y[order], Z[order], groups,
                                       nu=None, estimate_nu=True)

    ll0, g0, M0, _ = f(theta)
    num = np.empty_like(g0)
    h = 1e-6
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        num[j] = (f(tp)[0] - f(tm)[0]) / (2 * h)
    err = np.max(np.abs(g0 - num) / (1 + np.abs(num)))
    assert err < 1e-5
    assert np.abs(M0 - M0.T).max() < 1e-6
    assert np.linalg.eigvalsh(0.5 * (M0 + M0.T)).min() > -1e-6


def test_estimate_nu_matches_exact_gp_when_m_equals_n_minus_1():
    from scipy.spatial.distance import cdist
    from scaled_vecchia._matern_general import _matern_corr_general

    rng = np.random.default_rng(2)
    n, d = 50, 2
    X = rng.random((n, d))
    y = rng.standard_normal(n)
    theta = np.array([math.log(1.2), math.log(0.4), math.log(0.3),
                       math.log(1.75), math.log(0.02)])

    def exact_loglik():
        var, ranges = math.exp(theta[0]), np.exp(theta[1:1 + d])
        nu_val, nug = math.exp(theta[1 + d]), math.exp(theta[2 + d])
        r = cdist(X / ranges, X / ranges)
        Kmat = var * (_matern_corr_general(r, nu_val) + nug * np.eye(n))
        L = np.linalg.cholesky(Kmat)
        a = np.linalg.solve(L, y)
        return -0.5 * (n * LOG2PI + 2 * np.log(np.diag(L)).sum() + a @ a)

    m = n - 1
    order = maximin_order(X / np.exp(theta[1:1 + d]))
    nn = find_ordered_nn((X / np.exp(theta[1:1 + d]))[order], m)
    groups = _nn_groups(nn, m)

    ll_v, _, _, _ = vecchia_profile_loglik(theta, X[order], y[order], None,
                                            groups, nu=None, estimate_nu=True)
    assert abs(ll_v - exact_loglik()) < 1e-7
