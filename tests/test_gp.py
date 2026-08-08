import numpy as np
import pytest

from scaled_vecchia import ScaledVecchiaGP


def _sine_data(n, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sin(3 * X[:, 0]) + 0.1 * X[:, 1]
    return X, y


def test_fit_requires_matching_shapes():
    gp = ScaledVecchiaGP()
    X = np.random.rand(10, 2)
    y = np.random.rand(9)
    with pytest.raises(ValueError):
        gp.fit(X, y)


def test_predict_before_fit_raises():
    gp = ScaledVecchiaGP()
    with pytest.raises(RuntimeError):
        gp.predict(np.zeros((1, 2)))


def test_fit_predict_recovers_smooth_function():
    X, y = _sine_data(300, d=2, seed=0)
    gp = ScaledVecchiaGP(m_est=15, m_pred=30, nu=2.5, trend="constant",
                          n_est=300, random_state=0).fit(X, y)

    Xte, yte = _sine_data(200, d=2, seed=1)
    mu, sd = gp.predict(Xte, return_std=True)

    rmse = np.sqrt(np.mean((mu - yte) ** 2))
    assert rmse < 0.25
    assert np.all(sd >= 0)


def test_predict_return_var_consistent_with_return_std():
    X, y = _sine_data(150, seed=2)
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=150,
                          random_state=0).fit(X, y)
    Xte = np.random.default_rng(3).random((20, 2))

    mu1, sd = gp.predict(Xte, return_std=True)
    mu2, var = gp.predict(Xte, return_var=True)
    np.testing.assert_allclose(mu1, mu2)
    np.testing.assert_allclose(sd ** 2, var, rtol=1e-10)


def test_predict_mean_only_matches_full_call():
    X, y = _sine_data(120, seed=4)
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=120,
                          random_state=0).fit(X, y)
    Xte = np.random.default_rng(5).random((15, 2))

    mean_only = gp.predict(Xte)
    mean_full, _ = gp.predict(Xte, return_std=True)
    np.testing.assert_allclose(mean_only, mean_full)


def test_fixed_nugget_is_respected():
    X, y = _sine_data(120, seed=6)
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=120, nugget=1e-8,
                          random_state=0).fit(X, y)
    assert gp.nugget_ == pytest.approx(1e-8, rel=1e-6)


def test_relevance_and_ranges_are_positive():
    X, y = _sine_data(120, seed=7)
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=120,
                          random_state=0).fit(X, y)
    assert np.all(gp.ranges_ > 0)
    assert np.all(gp.relevance_ > 0)
    assert gp.variance_ > 0
    assert gp.nugget_ > 0


def test_predict_joint_mean_matches_marginal_predict():
    """The joint predictive mean should agree closely with the marginal
    predictive mean at the same locations (they use the same conditional
    Gaussian model, just factorised differently)."""
    X, y = _sine_data(200, seed=8)
    gp = ScaledVecchiaGP(m_est=15, m_pred=30, n_est=200,
                          random_state=0).fit(X, y)
    Xte = np.random.default_rng(9).random((25, 2))

    mu_marg = gp.predict(Xte)
    joint = gp.predict_joint(Xte, m=30, exact_var=True)
    np.testing.assert_allclose(mu_marg, joint["mean"], atol=0.05)
    assert np.all(joint["var"] >= 0)


def test_sample_joint_shape_and_finiteness():
    X, y = _sine_data(150, seed=10)
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=150,
                          random_state=0).fit(X, y)
    Xte = np.random.default_rng(11).random((12, 2))
    samples = gp.sample_joint(Xte, n_sim=25, m=20)
    assert samples.shape == (25, 12)
    assert np.all(np.isfinite(samples))


def test_summary_runs_after_fit():
    X, y = _sine_data(100, seed=12)
    gp = ScaledVecchiaGP(m_est=8, m_pred=15, n_est=100,
                          random_state=0).fit(X, y)
    text = gp.summary()
    assert "ScaledVecchiaGP" in text
    assert "loglik" in text


@pytest.mark.parametrize("trend", ["zero", "constant", "linear"])
def test_all_trend_options_fit_without_error(trend):
    X, y = _sine_data(100, seed=13)
    gp = ScaledVecchiaGP(m_est=8, m_pred=15, n_est=100, trend=trend,
                          random_state=0).fit(X, y)
    mu = gp.predict(X[:5])
    assert mu.shape == (5,)


def test_invalid_trend_raises():
    gp = ScaledVecchiaGP(trend="bogus")
    with pytest.raises(ValueError):
        gp.fit(np.random.rand(20, 2), np.random.rand(20))
