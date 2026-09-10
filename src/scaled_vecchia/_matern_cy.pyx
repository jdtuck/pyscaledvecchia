# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""
Optional Cython/OpenMP acceleration for the general (Bessel-based) Matern
covariance in `_matern_general.py`, used when the Matern smoothness `nu` is
estimated (`ScaledVecchiaGP(nu=None)`) or fixed at a non-half-integer value.

This is a drop-in accelerated implementation of
`_matern_general._block_cov_general`: same inputs, same outputs, same
maths, verified numerically identical (see `tests/test_likelihood.py` and
`tests/test_matern_cy.py`). It must be kept in sync with that function if
either changes.

Why this helps (and why it's not required)
--------------------------------------------
Profiling `_block_cov_general` shows ~85% of its time goes into
`scipy.special.kv` (the modified Bessel function of the second kind)
itself -- an already-compiled Fortran/C routine (AMOS algorithm), not a
Python-interpretation bottleneck. So the win here is not "Python is slow";
it is:

1. Fusing the whole per-block computation (distance, Bessel evaluation,
   analytic r-derivative, finite-difference nu-derivative, nugget) into a
   single native loop with no intermediate NumPy array allocation, calling
   `scipy.special.cython_special.kv` -- the *exact same* validated AMOS
   implementation `scipy.special.kv` uses, just reached directly at the C
   level with no per-call Python dispatch. This alone is a modest ~1.2-1.3x
   over the equivalent (already symmetry-optimized) NumPy/SciPy code.
2. `cython_special.kv` is declared `nogil`, so the loop over independent
   conditioning-set blocks can run under OpenMP (`prange`) across multiple
   threads -- something the pure NumPy/SciPy implementation cannot do at
   all (it is single-threaded). On a multi-core machine this is where most
   of the real speedup comes from; it scales with the number of threads
   given (up to the number of physical cores / memory bandwidth limits).

Because the underlying Bessel evaluations are delegated to SciPy's own
tested implementation rather than reimplemented, this carries no more
numerical-correctness risk than the pure-Python fallback -- it is purely a
speed optimization, not a different algorithm.

Building
--------
This extension is optional: `setup.py` attempts to compile it and degrades
gracefully (falling back to the pure NumPy/SciPy implementation in
`_matern_general.py`) if a C compiler or Cython is unavailable, or if
compilation otherwise fails. This never blocks installation of the rest of
the package, including the default (fixed nu in {0.5, 1.5, 2.5}, Numba
accelerated) fast path, which does not depend on this extension at all.
"""

import numpy as np

cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport exp, log, sqrt, M_LN2

cimport scipy.special.cython_special as cs

cnp.import_array()


cdef inline double _lgamma(double x) noexcept nogil:
    return cs.gammaln(x)


cdef inline double _sqdist(const double[:, :, ::1] Xb, const double[::1] ranges,
                            Py_ssize_t b, Py_ssize_t i, Py_ssize_t j,
                            Py_ssize_t D) noexcept nogil:
    cdef Py_ssize_t l
    cdef double s = 0.0
    cdef double diffl
    for l in range(D):
        diffl = (Xb[b, i, l] - Xb[b, j, l]) / ranges[l]
        s += diffl * diffl
    return s


def block_cov_general_cy(double[:, :, ::1] Xb not None,
                          double[::1] ranges not None,
                          double variance, double nugget, double nu,
                          bint derivs, nug_mask, bint estimate_nu,
                          double nu_fd_h, int num_threads):
    """Cython/OpenMP-accelerated equivalent of
    `_matern_general._block_cov_general`. See that function's docstring for
    the exact mathematical definitions -- this must stay numerically
    identical to it.

    Parameters
    ----------
    Xb, ranges, variance, nugget, nu, derivs, estimate_nu, nu_fd_h : as in
        `_block_cov_general`.
    nug_mask : None, or a (B, K) float64 ndarray (as in `_block_cov_general`).
    num_threads : number of OpenMP threads for the block loop (>=1).

    Returns
    -------
    Sigma : (B, K, K) ndarray
    dS    : (B, P, K, K) ndarray, or None if `derivs` is False.
    """
    cdef Py_ssize_t B = Xb.shape[0]
    cdef Py_ssize_t K = Xb.shape[1]
    cdef Py_ssize_t D = Xb.shape[2]
    cdef Py_ssize_t P = D + 2 + (1 if estimate_nu else 0)
    cdef double R_FLOOR = 1e-7

    cdef bint have_mask = nug_mask is not None
    cdef double[:, ::1] mask
    if have_mask:
        mask = np.ascontiguousarray(nug_mask, dtype=np.float64)

    cdef cnp.ndarray[cnp.float64_t, ndim=3] Sigma_arr = np.empty((B, K, K), dtype=np.float64)
    cdef double[:, :, ::1] Sigma = Sigma_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=4] dS_arr
    cdef double[:, :, :, ::1] dS
    if derivs:
        dS_arr = np.empty((B, P, K, K), dtype=np.float64)
        dS = dS_arr

    cdef double log_coef = (1.0 - nu) * M_LN2 - _lgamma(nu)
    cdef double sqrt2nu = sqrt(2.0 * nu)
    cdef double log_coef_p = 0.0, sqrt2nu_p = 0.0
    cdef double log_coef_m = 0.0, sqrt2nu_m = 0.0
    if estimate_nu:
        log_coef_p = (1.0 - (nu + nu_fd_h)) * M_LN2 - _lgamma(nu + nu_fd_h)
        sqrt2nu_p = sqrt(2.0 * (nu + nu_fd_h))
        log_coef_m = (1.0 - (nu - nu_fd_h)) * M_LN2 - _lgamma(nu - nu_fd_h)
        sqrt2nu_m = sqrt(2.0 * (nu - nu_fd_h))

    cdef Py_ssize_t b, i, j, l
    cdef Py_ssize_t nug_idx = P - 1
    cdef double s, r, rr, c, M, g, nugv, Mp, Mm, dMnu, diffl

    for b in prange(B, nogil=True, num_threads=num_threads, schedule='static'):
        for i in range(K):
            for j in range(i, K):
                s = _sqdist(Xb, ranges, b, i, j, D)
                r = sqrt(s)
                rr = r if r >= R_FLOOR else R_FLOOR
                c = sqrt2nu * rr
                if r < R_FLOOR:
                    M = 1.0
                else:
                    M = exp(log_coef + nu * log(c)) * cs.kv(nu, c)

                if have_mask:
                    nugv = nugget * mask[b, i] if i == j else 0.0
                else:
                    nugv = nugget if i == j else 0.0

                Sigma[b, i, j] = variance * (M + nugv)
                if i != j:
                    Sigma[b, j, i] = Sigma[b, i, j]

                if derivs:
                    dS[b, 0, i, j] = Sigma[b, i, j]
                    if i != j:
                        dS[b, 0, j, i] = Sigma[b, i, j]

                    if r < R_FLOOR:
                        g = 0.0
                    else:
                        g = -(sqrt2nu / rr) * exp(log_coef + nu * log(c)) * cs.kv(nu - 1.0, c)

                    for l in range(D):
                        diffl = (Xb[b, i, l] - Xb[b, j, l]) / ranges[l]
                        dS[b, 1 + l, i, j] = -variance * g * diffl * diffl
                        if i != j:
                            dS[b, 1 + l, j, i] = dS[b, 1 + l, i, j]

                    if estimate_nu:
                        if r < R_FLOOR:
                            Mp = 1.0
                            Mm = 1.0
                        else:
                            Mp = exp(log_coef_p + (nu + nu_fd_h) * log(sqrt2nu_p * rr)) * cs.kv(nu + nu_fd_h, sqrt2nu_p * rr)
                            Mm = exp(log_coef_m + (nu - nu_fd_h) * log(sqrt2nu_m * rr)) * cs.kv(nu - nu_fd_h, sqrt2nu_m * rr)
                        dMnu = (Mp - Mm) / (2.0 * nu_fd_h)
                        dS[b, 1 + D, i, j] = variance * nu * dMnu
                        if i != j:
                            dS[b, 1 + D, j, i] = dS[b, 1 + D, i, j]

                    dS[b, nug_idx, i, j] = variance * nugv
                    if i != j:
                        dS[b, nug_idx, j, i] = dS[b, nug_idx, i, j]

    if derivs:
        return Sigma_arr, dS_arr
    return Sigma_arr, None
