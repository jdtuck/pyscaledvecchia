"""
Tests for `scaledVecchia4mvBayes` / `MvBayesScaledVecchiaWrapper`, focused
on the MCMC-style repeated-call behavior it's designed for: many
`.predict(Xtest, ...)` calls against the *same* Xtest, each wanting a fresh
draw, with reproducibility from a fixed `random_state`.
"""

import numpy as np
import pytest

from scaled_vecchia import scaledVecchia4mvBayes


def _sine_data(n, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sin(3 * X[:, 0]) + 0.1 * X[:, 1]
    return X, y


def test_repeated_predict_same_xtest_gives_fresh_draws():
    X, y = _sine_data(150, seed=0)
    wrapper = scaledVecchia4mvBayes(X, y, nSamples=100, random_state=1,
                                     m_est=10, m_pred=20, n_est=150)
    Xte = np.random.default_rng(2).random((5, 2))

    draws = [wrapper.predict(Xte, idxSamples=[0])[0] for _ in range(5)]
    for i in range(1, len(draws)):
        assert not np.array_equal(draws[0], draws[i])


def test_reproducible_across_independently_constructed_wrappers():
    X, y = _sine_data(150, seed=3)
    Xte = np.random.default_rng(4).random((5, 2))

    w1 = scaledVecchia4mvBayes(X, y, nSamples=100, random_state=42,
                                m_est=10, m_pred=20, n_est=150)
    seq1 = [w1.predict(Xte, idxSamples=[0])[0] for _ in range(3)]

    w2 = scaledVecchia4mvBayes(X, y, nSamples=100, random_state=42,
                                m_est=10, m_pred=20, n_est=150)
    seq2 = [w2.predict(Xte, idxSamples=[0])[0] for _ in range(3)]

    for a, b in zip(seq1, seq2):
        np.testing.assert_array_equal(a, b)


def test_predict_shape_and_finiteness():
    X, y = _sine_data(120, seed=5)
    wrapper = scaledVecchia4mvBayes(X, y, nSamples=50, random_state=0,
                                     m_est=8, m_pred=15, n_est=120)
    Xte = np.random.default_rng(6).random((7, 2))

    out_all = wrapper.predict(Xte)
    assert out_all.shape == (50, 7)
    assert np.all(np.isfinite(out_all))

    out_some = wrapper.predict(Xte, idxSamples=[0, 1, 2])
    assert out_some.shape == (3, 7)


def test_cache_invalidates_when_xtest_changes():
    X, y = _sine_data(150, seed=7)
    wrapper = scaledVecchia4mvBayes(X, y, nSamples=100, random_state=0,
                                     m_est=10, m_pred=20, n_est=150)
    rng = np.random.default_rng(8)

    Xte1 = rng.random((5, 2))
    s1 = wrapper.predict(Xte1, idxSamples=[0])
    key1 = wrapper._joint_cache_key

    Xte2 = rng.random((6, 2))   # different shape -> definitely a cache miss
    s2 = wrapper.predict(Xte2, idxSamples=[0])
    key2 = wrapper._joint_cache_key

    assert key1 != key2
    assert s1.shape == (1, 5)
    assert s2.shape == (1, 6)

    # switching back to Xte1 should rebuild the cache and still be correct
    s3 = wrapper.predict(Xte1, idxSamples=[0])
    assert s3.shape == (1, 5)


def test_resid_sd_attribute_present_and_positive():
    X, y = _sine_data(100, seed=9)
    wrapper = scaledVecchia4mvBayes(X, y, nSamples=20, random_state=0,
                                     m_est=8, m_pred=15, n_est=100)
    assert wrapper.samples.residSD.shape == (20,)
    assert np.all(wrapper.samples.residSD > 0)


def test_invalid_nsamples_raises():
    X, y = _sine_data(50, seed=10)
    with pytest.raises(ValueError):
        scaledVecchia4mvBayes(X, y, nSamples=0)
    with pytest.raises(ValueError):
        scaledVecchia4mvBayes(X, y, nSamples=-5)
