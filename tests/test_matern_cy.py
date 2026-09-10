"""
Tests for the optional Cython/OpenMP backend (`_matern_cy`).

If the extension wasn't built (no C compiler / Cython available at install
time), these tests are skipped entirely -- the rest of the test suite
already covers correctness of the pure-Python fallback, which remains the
default whenever this extension isn't present. When it *is* present, these
tests check it against the pure-Python engine directly (bypassing whichever
one `_block_cov_general` would normally dispatch to), across the same
settings the rest of the suite already validates the pure-Python engine
against (derivatives on/off, nu estimation on/off, with/without a nugget
mask, thread-count invariance).
"""

import numpy as np
import pytest

from scaled_vecchia._matern_general import (
    HAVE_CYTHON_BACKEND,
    _block_cov_general_cython,
    _block_cov_general_numpy,
)

pytestmark = pytest.mark.skipif(
    not HAVE_CYTHON_BACKEND,
    reason="optional Cython acceleration extension was not built",
)


def _random_block(seed, B=40, K=17, d=4):
    rng = np.random.default_rng(seed)
    Xb = rng.random((B, K, d))
    ranges = 0.2 + 0.5 * rng.random(d)
    return Xb, ranges


@pytest.mark.parametrize("derivs", [True, False])
@pytest.mark.parametrize("estimate_nu", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_cython_matches_numpy_engine(derivs, estimate_nu, use_mask):
    Xb, ranges = _random_block(0)
    B, K, d = Xb.shape
    variance, nugget, nu = 1.3, 0.02, 1.75

    if estimate_nu and not derivs:
        pytest.skip("estimate_nu only adds a derivative row; nothing to check without derivs")

    nug_mask = None
    if use_mask:
        rng = np.random.default_rng(1)
        nug_mask = (rng.random((B, K)) > 0.3).astype(float)

    Sig_py, dS_py = _block_cov_general_numpy(
        Xb, ranges, variance, nugget, nu, derivs=derivs,
        nug_mask=nug_mask, estimate_nu=estimate_nu)
    Sig_cy, dS_cy = _block_cov_general_cython(
        Xb, ranges, variance, nugget, nu, derivs=derivs,
        nug_mask=nug_mask, estimate_nu=estimate_nu, num_threads=1)

    np.testing.assert_allclose(Sig_py, Sig_cy, atol=1e-10, rtol=1e-10)
    if derivs:
        np.testing.assert_allclose(dS_py, dS_cy, atol=1e-9, rtol=1e-9)
    else:
        assert dS_py is None and dS_cy is None


def test_cython_result_invariant_to_thread_count():
    Xb, ranges = _random_block(2, B=200, K=21, d=5)
    variance, nugget, nu = 0.9, 0.05, 2.1

    Sig1, dS1 = _block_cov_general_cython(
        Xb, ranges, variance, nugget, nu, derivs=True,
        estimate_nu=True, num_threads=1)
    for nt in (2, 4, 8):
        Sig_n, dS_n = _block_cov_general_cython(
            Xb, ranges, variance, nugget, nu, derivs=True,
            estimate_nu=True, num_threads=nt)
        np.testing.assert_allclose(Sig1, Sig_n, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(dS1, dS_n, atol=1e-12, rtol=1e-12)


def test_cython_symmetry_of_output():
    """Sigma (and each derivative slice) must be exactly symmetric."""
    Xb, ranges = _random_block(3, B=10, K=13, d=3)
    Sig, dS = _block_cov_general_cython(
        Xb, ranges, 1.1, 0.01, 1.6, derivs=True, estimate_nu=True, num_threads=1)
    np.testing.assert_array_equal(Sig, Sig.transpose(0, 2, 1))
    for p in range(dS.shape[1]):
        np.testing.assert_array_equal(dS[:, p], dS[:, p].transpose(0, 2, 1))


def test_cython_reduces_to_half_integer_closed_form():
    """Cross-check against the closed-form half-integer Matern (a completely
    independent implementation) as an extra sanity check on the compiled
    extension, not just against the pure-Python general engine."""
    from scaled_vecchia._covariance import _matern_corr

    Xb, ranges = _random_block(4, B=15, K=11, d=2)
    for nu in (0.5, 1.5, 2.5):
        Sig, _ = _block_cov_general_cython(
            Xb, ranges, 1.0, 0.0, nu, derivs=False, num_threads=1)
        U = Xb / ranges
        diff = U[:, :, None, :] - U[:, None, :, :]
        r = np.sqrt((diff ** 2).sum(-1))
        expected = _matern_corr(r, nu)
        np.testing.assert_allclose(Sig, expected, atol=1e-10, rtol=1e-8)


def test_gp_end_to_end_with_cython_backend_active():
    """The full estimate_nu fit/predict path, exercised specifically through
    the compiled backend (the default whenever it's built)."""
    from scaled_vecchia import ScaledVecchiaGP

    rng = np.random.default_rng(5)
    n, d = 250, 2
    X = rng.random((n, d))
    y = np.sin(3 * X[:, 0]) + 0.1 * X[:, 1]

    gp = ScaledVecchiaGP(m_est=15, m_pred=30, nu=None, n_est=250,
                          random_state=0).fit(X, y)
    assert gp._estimate_nu is True
    assert gp.nu_ > 0

    Xte = rng.random((30, d))
    mu, sd = gp.predict(Xte, return_std=True)
    assert np.all(np.isfinite(mu)) and np.all(sd >= 0)
    assert "Cython/OpenMP" in gp.summary()
