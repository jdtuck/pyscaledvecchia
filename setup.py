"""
Builds the optional `scaled_vecchia._matern_cy` Cython/OpenMP extension
(see `src/scaled_vecchia/_matern_cy.pyx` for what it does and why).

This extension is entirely optional: if Cython isn't installed, no C
compiler is available, or compilation fails for any other reason, the
build degrades gracefully -- a warning is printed and the package still
installs successfully as pure Python/NumPy/SciPy/Numba, with
`ScaledVecchiaGP(nu=None)` (or any non-half-integer fixed `nu`) simply
falling back to the slower pure-Python general-Matern implementation.
Nothing else in the package (including the default, Numba-accelerated,
fixed nu in {0.5, 1.5, 2.5} fast path) depends on this extension at all.

OpenMP: enabled on Linux and Windows. Skipped on macOS, since Apple's
shipped Clang does not support -fopenmp out of the box (it requires a
separate libomp install); on macOS the extension still builds and runs,
just single-threaded, which is still faster than the pure-Python path
because scipy.special.cython_special.kv is called directly instead of
through scipy's Python-level ufunc dispatch, and the whole per-block
computation is fused into one native loop instead of several separate
NumPy array passes.
"""

import platform
import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

_EXT_NAME = "scaled_vecchia._matern_cy"
_PYX_PATH = "src/scaled_vecchia/_matern_cy.pyx"

_WARN_PREFIX = (
    "scaled-vecchia: could not build the optional Cython acceleration "
    "extension for general/estimated-nu Matern covariance"
)
_WARN_SUFFIX = (
    "; falling back to the pure NumPy/SciPy implementation. This does NOT "
    "affect installation of the rest of the package or the default "
    "(fixed nu in {0.5, 1.5, 2.5}, Numba-accelerated) fast path."
)


def _compile_args():
    system = platform.system()
    if system == "Windows":
        return ["/O2", "/openmp"], []
    if system == "Darwin":
        # Stock Apple Clang does not support -fopenmp; skip it rather than
        # fail the build. The extension still compiles and runs (just
        # single-threaded), which is still a real speedup over pure Python.
        return ["-O3"], []
    return ["-O3", "-fopenmp"], ["-fopenmp"]


def _make_ext_modules():
    try:
        from Cython.Build import cythonize
    except ImportError as e:
        warnings.warn(f"{_WARN_PREFIX} (Cython is not installed: {e}){_WARN_SUFFIX}")
        return []

    try:
        import numpy as np
    except ImportError as e:
        warnings.warn(f"{_WARN_PREFIX} (NumPy is not available at build time: {e}){_WARN_SUFFIX}")
        return []

    compile_args, link_args = _compile_args()
    ext = Extension(
        _EXT_NAME,
        [_PYX_PATH],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
    try:
        return cythonize(
            [ext],
            compiler_directives={"language_level": "3"},
            # Don't let a stale .c from a previous failed/partial build hide
            # source changes; and keep this fast+quiet for normal installs.
            force=True,
            quiet=True,
        )
    except Exception as e:  # cythonize() itself can raise (e.g. cimport
        # failures against an incompatible scipy version) -- never let that
        # abort the whole install.
        warnings.warn(f"{_WARN_PREFIX} (Cython compilation step failed: {e}){_WARN_SUFFIX}")
        return []


class optional_build_ext(_build_ext):
    """Never let a failure to build this optional extension fail the
    overall `pip install` -- catch compiler/linker errors per-extension and
    warn instead."""

    def run(self):
        try:
            super().run()
        except Exception as e:
            warnings.warn(f"{_WARN_PREFIX} (build_ext failed: {e}){_WARN_SUFFIX}")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            warnings.warn(f"{_WARN_PREFIX} (compiling {ext.name} failed: {e}){_WARN_SUFFIX}")


setup(
    ext_modules=_make_ext_modules(),
    cmdclass={"build_ext": optional_build_ext},
)
