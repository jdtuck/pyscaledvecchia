"""
Standalone mvBayes-compatible wrapper for scaled_vecchia.ScaledVecchiaGP.

Behavior
--------
- Fits a ScaledVecchiaGP model to a univariate response y.
- Returns joint posterior samples of the latent mean function via predict(Xtest, idxSamples=...).
- Does NOT add nugget noise to those predictive samples.
- Uses the fitted nugget to define .samples.residSD for mvBayes.
- Returns predictions in shape (n_samples, n_obs), compatible with mvBayes.

Default synthetic posterior sample count:
    nSamples = 1000

Performance
-----------
In an MCMC/Bayesian-calibration loop, `.predict(Xtest, ...)` is typically
called many times against the *same* `Xtest` (the emulator's design is
fixed for the whole chain; only which/how many draws are requested varies).
This wrapper caches the expensive part of joint prediction (ordering,
covariance evaluation, sparse factorization -- see
`ScaledVecchiaGP.prepare_joint`) the first time a given `Xtest` is seen and
reuses it on subsequent calls with the same `Xtest`, needing only a cheap
sparse triangular solve per call thereafter. It also draws from a
persistent per-model random stream by default (seeded once from
`random_state`, if given), so repeated calls give *fresh* draws -- not the
same one repeated -- while the whole sequence stays reproducible from that
seed.
"""

import numpy as np

from .gp import ScaledVecchiaGP


class _MvBayesScaledVecchiaSamples:
    """Simple container for mvBayes-compatible posterior sample attributes."""
    pass


class MvBayesScaledVecchiaWrapper:
    """
    mvBayes-compatible wrapper around scaled_vecchia.ScaledVecchiaGP.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix.
    y : np.ndarray
        Univariate response.
    nSamples : int, default=1000
        Number of joint latent mean-function samples to use as synthetic posterior draws.
    random_state : int or None, default=None
        Seeds the model's persistent random stream (see `ScaledVecchiaGP._get_rng`):
        fixes the whole sequence of draws across repeated `.predict()` calls for
        reproducibility, without forcing every individual call to return the
        same sample.
    **kwargs
        Additional keyword arguments passed to ScaledVecchiaGP(...) constructor.
    """

    def __init__(self, X, y, nSamples=1000, random_state=None, **kwargs):
        y = np.asarray(y)
        if y.ndim != 1:
            y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array or coercible to 1D.")

        if not isinstance(nSamples, int) or nSamples <= 0:
            raise ValueError("nSamples must be a positive integer.")

        self.nSamples = nSamples
        self.random_state = random_state

        gp_kwargs = dict(kwargs)
        if random_state is not None and "random_state" not in gp_kwargs:
            gp_kwargs["random_state"] = random_state

        self.model = ScaledVecchiaGP(**gp_kwargs)
        self.model.fit(X, y)

        self.samples = _MvBayesScaledVecchiaSamples()

        # Residual SD on original response scale:
        # standardized-scale residual variance is variance_ * nugget_
        # original-scale SD multiplies by model._ysd
        residSD_scalar = np.sqrt(self.model.variance_ * self.model.nugget_) * self.model._ysd
        self.samples.residSD = np.repeat(residSD_scalar, self.nSamples)

        # Cache for the (expensive) joint-prediction setup at a fixed Xtest;
        # see class/module docstring. Invalidated whenever Xtest changes.
        self._joint_cache = None
        self._joint_cache_key = None

    @staticmethod
    def _xtest_key(Xtest):
        return (Xtest.shape, hash(Xtest.tobytes()))

    def predict(self, Xtest, idxSamples=None):
        """
        Return joint samples of the latent mean function (no nugget noise added).

        Parameters
        ----------
        Xtest : array-like
            Test predictors.
        idxSamples : None or array-like of int
            If None, return self.nSamples draws.
            If provided, generate only len(idxSamples) draws and return them.

        Returns
        -------
        np.ndarray
            Shape (n_samples_selected, n_obs), compatible with mvBayes.
        """
        if idxSamples is None:
            n_draws = self.nSamples
        else:
            idxSamples = np.asarray(idxSamples)
            if idxSamples.ndim == 0:
                idxSamples = idxSamples.reshape(1)
            n_draws = len(idxSamples)

        Xtest = np.ascontiguousarray(np.atleast_2d(Xtest), dtype=float)
        key = self._xtest_key(Xtest)
        if self._joint_cache is None or self._joint_cache_key != key:
            self._joint_cache = self.model.prepare_joint(Xtest)
            self._joint_cache_key = key

        # random_state=None here (rather than self.random_state) is
        # deliberate: it draws the *next* values from the model's
        # persistent stream (already seeded from self.random_state at
        # construction) instead of resetting to the same seed every call,
        # which is what gives every .predict() call a fresh sample.
        samples = self._joint_cache.sample(n_sim=n_draws, random_state=None)

        samples = np.asarray(samples)  # expected shape: (n_samples, n_obs)

        if samples.ndim != 2:
            raise ValueError(
                f"Expected joint samples with 2 dimensions, got shape {samples.shape}."
            )

        return samples


def scaledVecchia4mvBayes(X, y, **kwargs):
    """
    Factory function for use as mvBayes(..., bayesModel=...).

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix.
    y : np.ndarray
        Univariate response.
    **kwargs
        Additional keyword arguments passed to MvBayesScaledVecchiaWrapper.
        These include:
          - nSamples
          - random_state
          - any valid ScaledVecchiaGP constructor kwargs

    Returns
    -------
    MvBayesScaledVecchiaWrapper
        mvBayes-compatible fitted model object.
    """
    return MvBayesScaledVecchiaWrapper(X, y, **kwargs)
    
