"""
scaled_vecchia
==============

A NumPy/SciPy implementation of

    Katzfuss, M., Guinness, J., & Lawrence, E. (2020/2022).
    "Scaled Vecchia approximation for fast computer-model emulation."
    arXiv:2005.00386 / SIAM-ASA J. Uncertainty Quantification 10(2).

See the package README for a full description of the method. Quick start::

    from scaled_vecchia import ScaledVecchiaGP
    gp = ScaledVecchiaGP(m_est=30, m_pred=140, nu=2.5).fit(X_train, y_train)
    mean, sd = gp.predict(X_test, return_std=True)
    draws    = gp.sample_joint(X_path, n_sim=100)   # joint paths
"""

from .gp import ScaledVecchiaGP
from .likelihood import vecchia_profile_loglik
from .ordering import find_ordered_nn, maximin_order
from .scaledVecchia4mvBayes import scaledVecchia4mvBayes

__all__ = [
    "ScaledVecchiaGP",
    "maximin_order",
    "find_ordered_nn",
    "vecchia_profile_loglik",
    "scaledVecchia4mvBayes",
]

__version__ = "0.1.0"
