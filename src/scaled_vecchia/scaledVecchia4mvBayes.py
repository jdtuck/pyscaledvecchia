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
"""

import numpy as np

from scaled_vecchia import ScaledVecchiaGP


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
        Random seed passed to joint predictive sampling unless overridden in model kwargs.
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

    def predict(self, Xtest, idxSamples=None):
        """
        Return joint samples of the latent mean function (no nugget noise added).

        Parameters
        ----------
        Xtest : array-like
            Test predictors.
        idxSamples : None or array-like of int
            Posterior sample indices to retain. If None, all synthetic posterior draws are returned.

        Returns
        -------
        np.ndarray
            Shape (n_samples_selected, n_obs), compatible with mvBayes.
        """
        samples = self.model.sample_joint(
            Xtest,
            n_sim=self.nSamples,
            random_state=self.random_state,
        )

        samples = np.asarray(samples)  # expected shape: (n_samples, n_obs)

        if samples.ndim != 2:
            raise ValueError(
                f"Expected joint samples with 2 dimensions, got shape {samples.shape}."
            )

        if idxSamples is not None:
            idxSamples = np.asarray(idxSamples, dtype=int)
            samples = samples[idxSamples, :]

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
    
