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


# ---------------------------------------------------------------------------
# Matern smoothness (nu): fixed general value, and estimation
# ---------------------------------------------------------------------------

def test_fixed_noninteger_nu_fits_without_error():
    """nu fixed at a value with no closed form should transparently use the
    general Bessel-based covariance path."""
    X, y = _sine_data(150, seed=20)
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=150, nu=1.75,
                          random_state=0).fit(X, y)
    assert gp.nu_ == pytest.approx(1.75)
    mu = gp.predict(X[:5])
    assert mu.shape == (5,) and np.all(np.isfinite(mu))


def test_estimate_nu_recovers_reasonable_value_on_simulated_matern_data():
    """Simulate data from a *known* general-nu Matern GP and check the
    estimated nu lands in a sane neighbourhood of the truth. nu is a
    famously hard parameter to pin down exactly with finite data (see e.g.
    Zhang 2004 on infill asymptotics), so this only checks a wide, sane
    range rather than tight recovery."""
    from scipy.spatial.distance import cdist

    from scaled_vecchia._matern_general import _matern_corr_general

    rng = np.random.default_rng(0)
    n, d = 500, 2
    X = rng.random((n, d))
    true_nu, true_range, true_var, true_nug = 1.75, np.array([0.25, 0.4]), 1.5, 0.02

    r = cdist(X / true_range, X / true_range)
    K = true_var * (_matern_corr_general(r, true_nu) + true_nug * np.eye(n))
    L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
    y = L @ rng.standard_normal(n)

    gp = ScaledVecchiaGP(m_est=25, m_pred=50, nu=None, n_est=500, trend="zero",
                          var_correction=False, random_state=0).fit(X, y)

    assert gp._estimate_nu is True
    assert 0.5 < gp.nu_ < 5.0          # sane neighbourhood of the true 1.75
    assert gp.nugget_ == pytest.approx(true_nug, rel=0.5)
    # the smaller true range (x0) should be recovered as more relevant
    assert gp.relevance_[0] > gp.relevance_[1]


def test_estimate_nu_predict_runs_and_is_finite():
    X, y = _sine_data(200, seed=21)
    gp = ScaledVecchiaGP(m_est=15, m_pred=30, n_est=200, nu=None,
                          random_state=0).fit(X, y)
    Xte = np.random.default_rng(22).random((30, 2))
    mu, sd = gp.predict(Xte, return_std=True)
    assert np.all(np.isfinite(mu)) and np.all(sd >= 0)
    assert gp.nu_ > 0


def test_nu_property_matches_fixed_setting_when_not_estimated():
    X, y = _sine_data(100, seed=23)
    gp = ScaledVecchiaGP(m_est=8, m_pred=15, n_est=100, nu=2.5,
                          random_state=0).fit(X, y)
    assert gp._estimate_nu is False
    assert gp.nu_ == 2.5


# ---------------------------------------------------------------------------
# Joint prediction performance/correctness fixes: order_obs caching,
# prepare_joint + JointPredictiveCache, and the persistent-RNG-by-default fix
# ---------------------------------------------------------------------------

def test_repeated_unseeded_sample_joint_gives_fresh_draws():
    """random_state=None must give a *different* draw each call (previously
    it silently repeated the same draw every time -- see prepare_joint's
    and _get_rng's docstrings)."""
    X, y = _sine_data(200, seed=30)
    gp = ScaledVecchiaGP(m_est=15, m_pred=30, n_est=200, random_state=0).fit(X, y)
    Xte = np.random.default_rng(31).random((5, 2))

    draws = [gp.sample_joint(Xte, n_sim=1, m=30) for _ in range(5)]
    for i in range(1, len(draws)):
        assert not np.array_equal(draws[0], draws[i])


def test_explicit_random_state_is_reproducible_and_independent_of_stream():
    X, y = _sine_data(150, seed=32)
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=150, random_state=0).fit(X, y)
    Xte = np.random.default_rng(33).random((5, 2))

    a = gp.sample_joint(Xte, n_sim=1, m=20, random_state=7)
    gp.sample_joint(Xte, n_sim=1, m=20)   # advances the persistent stream
    b = gp.sample_joint(Xte, n_sim=1, m=20, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_refitting_resets_the_persistent_random_stream_reproducibly():
    X, y = _sine_data(150, seed=34)
    Xte = np.random.default_rng(35).random((5, 2))

    gp1 = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=150, random_state=0).fit(X, y)
    seq1 = [gp1.sample_joint(Xte, n_sim=1, m=20) for _ in range(3)]

    gp2 = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=150, random_state=0).fit(X, y)
    seq2 = [gp2.sample_joint(Xte, n_sim=1, m=20) for _ in range(3)]

    for a, b in zip(seq1, seq2):
        np.testing.assert_array_equal(a, b)


def test_order_obs_cache_does_not_change_results_across_different_xstar():
    """Repeated predict_joint calls at *different* Xstar must still be
    correct after order_obs (the training-data ordering) is cached -- it
    must not accidentally leak stale state."""
    X, y = _sine_data(250, seed=36)
    gp = ScaledVecchiaGP(m_est=15, m_pred=30, n_est=250, random_state=0).fit(X, y)
    rng = np.random.default_rng(37)

    for _ in range(3):
        Xte = rng.random((10, 2))
        out = gp.predict_joint(Xte, m=30, exact_var=True)
        mu_direct, var_direct = gp.predict(Xte, m=30, return_var=True)
        # joint marginal mean should closely track the independent marginal
        np.testing.assert_allclose(out["mean"], mu_direct, atol=0.2)
        assert np.all(out["var"] >= 0)


def test_order_obs_cache_invalidated_by_refit():
    """Directly checks the caching mechanism (not just predicted values,
    which can look similar across two fits of the same deterministic
    function regardless of whether the cache is stale)."""
    X1, y1 = _sine_data(150, seed=38)
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=150, random_state=0).fit(X1, y1)
    Xte = np.random.default_rng(39).random((5, 2))
    gp.predict_joint(Xte, m=20)   # populates the cache
    assert gp._order_obs_cache is not None
    assert gp._order_obs_cache.shape == (150,)

    X2, y2 = _sine_data(180, seed=40)   # different size -> unambiguous check
    gp.fit(X2, y2)
    assert gp._order_obs_cache is None   # invalidated immediately by fit()

    gp.predict_joint(Xte, m=20)          # repopulated fresh, matching new data
    assert gp._order_obs_cache.shape == (180,)


def test_prepare_joint_mean_and_var_match_predict_joint():
    X, y = _sine_data(200, seed=41)
    gp = ScaledVecchiaGP(m_est=15, m_pred=30, n_est=200, random_state=0).fit(X, y)
    Xte = np.random.default_rng(42).random((15, 2))

    out = gp.predict_joint(Xte, m=30, exact_var=True)
    cache = gp.prepare_joint(Xte, m=30)
    np.testing.assert_allclose(out["mean"], cache.mean, atol=1e-10)
    np.testing.assert_allclose(out["var"], cache.var, atol=1e-10)


def test_prepare_joint_sample_gives_fresh_draws_matching_distribution():
    X, y = _sine_data(300, seed=43)
    gp = ScaledVecchiaGP(m_est=15, m_pred=30, n_est=300, random_state=0).fit(X, y)
    Xte = np.random.default_rng(44).random((10, 2))

    cache = gp.prepare_joint(Xte, m=30)
    d1 = cache.sample(n_sim=500)
    d2 = cache.sample(n_sim=500)
    assert not np.array_equal(d1, d2)
    np.testing.assert_allclose(d1.mean(0), cache.mean, atol=0.2)
    np.testing.assert_allclose(d1.var(0), cache.var, rtol=0.4)


def test_prepare_joint_becomes_stale_after_refit_is_a_known_limitation():
    """A cache from `prepare_joint` reflects the model at the time it was
    prepared and is not silently updated by a later `fit()` call -- calling
    `prepare_joint` again after refitting is required to see the new fit.
    Uses two genuinely different functions (not just different samples of
    the same function) so the two fits are guaranteed to disagree, rather
    than relying on estimation noise to differ measurably."""
    rng = np.random.default_rng(45)
    X1 = rng.random((150, 2))
    y1 = np.sin(3 * X1[:, 0])
    gp = ScaledVecchiaGP(m_est=10, m_pred=20, n_est=150, random_state=0).fit(X1, y1)
    Xte = np.random.default_rng(46).random((5, 2))
    stale_cache = gp.prepare_joint(Xte, m=20)

    X2 = rng.random((150, 2))
    y2 = -5.0 * X2[:, 0] + 3.0   # unrelated linear function, far from sin(3x0)
    gp.fit(X2, y2)
    fresh_cache = gp.prepare_joint(Xte, m=20)

    assert not np.allclose(stale_cache.mean, fresh_cache.mean, atol=0.3)
