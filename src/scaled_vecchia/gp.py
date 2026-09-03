"""The scaled Vecchia GP emulator (Katzfuss, Guinness & Lawrence, 2020/2022).

The method fits an anisotropic (ARD) Gaussian-process emulator to a large
computer experiment by

  1. scaling each input dimension by its estimated range parameter,
     x~ = (x_1/lambda_1, ..., x_d/lambda_d)                         [Sec. 3.1]
  2. computing a maximin ordering and nearest-neighbour conditioning sets
     *in the scaled space*,                                         [Sec. 3.1]
  3. maximising the resulting Vecchia loglikelihood with Fisher scoring,
     profiling out the linear-mean coefficients beta by GLS,        [Sec. 3.2]
     re-doing (1)-(2) at iterations k = 2, 4, 8, 16, ...
  4. predicting with an "observed-first" maximin ordering of the combined
     training + test inputs, which yields a *joint* Gaussian predictive
     distribution with a sparse inverse Cholesky factor,            [Sec. 3.3]
  5. optionally rescaling predictive variances by a factor b estimated on a
     held-out inner split (variance correction).                    [Sec. 3.4]

Everything scales as O(n m^3) for estimation and O(n* m*^3) for prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from scipy.spatial import cKDTree

from ._covariance import _backsolve_LT, _batch_chol, _block_cov, _chunks
from .likelihood import vecchia_profile_loglik
from .optimize import _fisher_scoring
from .ordering import _nn_groups, find_ordered_nn, maximin_order

__all__ = ["ScaledVecchiaGP"]


@dataclass
class ScaledVecchiaGP:
    """Scaled Vecchia GP emulator (Katzfuss, Guinness & Lawrence, 2020).

    Parameters
    ----------
    m_est   : conditioning-set size for likelihood evaluation (paper default 30)
    m_pred  : conditioning-set size for prediction            (paper default 140)
    n_est   : subsample size used for parameter estimation    (paper default 5000)
    nu      : Matern smoothness. A fixed positive float (0.5, 1.5, 2.5 use fast
              Numba-compiled closed forms; any other value falls back to the
              general Bessel-based Matern, which is slower per block), or
              `None` to *estimate* nu jointly with the other covariance
              parameters via Fisher scoring (Sec. 3.5 of the paper). The
              paper's own reference implementation does exactly this when nu
              is left unspecified, starting from an initial smoothness of
              3.5; estimating nu always uses the (non-Numba) Bessel-based
              covariance, since the derivative of a Bessel function with
              respect to its order has no closed form and the half-integer
              fast paths only exist at fixed nu.
    trend   : 'constant' (psi(x) = 1), 'linear' (psi(x) = (1, x')'), or 'zero'
    nugget  : None -> estimate the relative nugget; a float -> hold it fixed
              (use e.g. 1e-8 for a deterministic computer model)
    var_correction : estimate the predictive-variance inflation factor b of
              Sec. 3.4 on an inner train/test split
    lambda_max : ranges above this are treated as infinite (variable selection,
              Sec. 3.2); set to np.inf to disable
    """

    m_est: int = 30
    m_pred: int = 140
    n_est: int = 5000
    nu: float | None = 2.5
    trend: str = "constant"
    nugget: float | None = None
    var_penalty: float = 1.0
    var_correction: bool = True
    lambda_max: float = 1e3
    max_iter: int = 40
    tol: float = 1e-4
    random_state: int | None = 0
    verbose: bool = False

    # fitted state -----------------------------------------------------------
    X_: np.ndarray = field(default=None, init=False, repr=False)
    y_: np.ndarray = field(default=None, init=False, repr=False)
    theta_: np.ndarray = field(default=None, init=False, repr=False)
    beta_: np.ndarray = field(default=None, init=False, repr=False)
    b_: float = field(default=1.0, init=False)
    d_: int = field(default=None, init=False, repr=False)
    _estimate_nu: bool = field(default=False, init=False, repr=False)
    _nu_fit: float = field(default=None, init=False, repr=False)

    # ---------------- basis / scaling helpers ------------------------------
    def _design(self, X):
        if self.trend == "zero":
            return np.zeros((X.shape[0], 0))
        if self.trend == "constant":
            return np.ones((X.shape[0], 1))
        if self.trend == "linear":
            return np.column_stack([np.ones(X.shape[0]), X])
        raise ValueError("trend must be 'zero', 'constant' or 'linear'")

    @property
    def variance_(self):
        return math.exp(self.theta_[0])

    @property
    def ranges_(self):
        return np.exp(self.theta_[1:1 + self.d_])

    @property
    def relevance_(self):
        """1 / lambda_l -- how strongly input l affects the response (Sec. 3.1)."""
        return 1.0 / self.ranges_

    @property
    def nu_(self):
        """Matern smoothness actually used: the fixed `nu`, or the fitted
        value if `nu=None` requested estimation."""
        return self._nu_fit

    @property
    def nugget_(self):
        return math.exp(self.theta_[-1])

    def _clip_theta(self, theta, d, estimate_nu):
        hi = math.log(self.lambda_max) if np.isfinite(self.lambda_max) else np.inf
        theta[1:1 + d] = np.clip(theta[1:1 + d], math.log(1e-8), hi)
        if estimate_nu:
            # Keep the Bessel evaluations well-conditioned and the smoothness
            # in a practically identifiable range (very large nu is close to
            # indistinguishable from a squared-exponential kernel).
            theta[1 + d] = np.clip(theta[1 + d], math.log(0.3), math.log(10.0))
        theta[-1] = np.clip(theta[-1], math.log(1e-12), math.log(1e3))
        return theta

    # ---------------- fitting ----------------------------------------------
    def fit(self, X, y):
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y have incompatible shapes")

        estimate_nu = self.nu is None
        self._estimate_nu = estimate_nu
        self.d_ = d

        # Standardise the input box to [0,1]^d so that 1/lambda_l is comparable
        # across dimensions, and centre/scale y for numerical conditioning.
        self._lo = X.min(0)
        self._span = np.where(X.max(0) - X.min(0) > 0, X.max(0) - X.min(0), 1.0)
        self._ymu = y.mean()
        self._ysd = y.std() if y.std() > 0 else 1.0

        self.X_ = (X - self._lo) / self._span
        self.y_ = (y - self._ymu) / self._ysd

        rng = np.random.default_rng(self.random_state)
        if self.n_est is not None and n > self.n_est:
            sub = rng.choice(n, self.n_est, replace=False)
        else:
            sub = np.arange(n)
        Xe, ye = self.X_[sub], self.y_[sub]
        Ze = self._design(Xe)
        m = min(self.m_est, len(sub) - 1)

        # ---- initial values -----------------------------------------------
        P0 = d + 2 + (1 if estimate_nu else 0)
        theta = np.empty(P0)
        theta[0] = math.log(max(ye.var(), 1e-8))
        theta[1:1 + d] = math.log(0.2 * math.sqrt(d))     # ranges in the unit box
        fixed_nug = self.nugget is not None
        if estimate_nu:
            theta[1 + d] = math.log(1.5)                  # initial smoothness guess
        theta[-1] = math.log(self.nugget if fixed_nug else 0.1)
        theta = self._clip_theta(theta, d, estimate_nu)

        # ---- mutable Vecchia structure (refreshed by the reorder hook) -----
        state = {}

        def rebuild(th):
            ranges = np.exp(th[1:1 + d])
            Xs = Xe / ranges
            order = maximin_order(Xs)
            nn = find_ordered_nn(Xs[order], m)
            state["order"] = order
            state["groups"] = _nn_groups(nn, m)
            state["Xo"] = Xe[order]
            state["yo"] = ye[order]
            state["Zo"] = Ze[order]

        rebuild(theta)
        last_ranges = np.exp(theta[1:1 + d]).copy()

        def objective(th):
            th = self._clip_theta(np.array(th, dtype=float), d, estimate_nu)
            if fixed_nug:
                th[-1] = math.log(self.nugget)
            ll, g, M, beta = vecchia_profile_loglik(
                th, state["Xo"], state["yo"], state["Zo"], state["groups"],
                self.nu, estimate_nu=estimate_nu, need_grad=True,
                var_penalty=self.var_penalty,
                log_var_target=math.log(max(ye.var(), 1e-8)),
            )
            if g is not None and fixed_nug:      # freeze the nugget direction
                g = g.copy(); g[-1] = 0.0
                M = M.copy(); M[-1, :] = 0.0; M[:, -1] = 0.0; M[-1, -1] = 1.0
            return ll, g, M, beta

        def reorder_hook(th, k):
            nonlocal last_ranges
            r = np.exp(self._clip_theta(np.array(th), d, estimate_nu)[1:1 + d])
            if np.max(np.abs(np.log(r / last_ranges))) < 1e-3:
                return False
            if self.verbose:
                print(f"    [reorder at iter {k}] ranges = "
                      + np.array2string(r, precision=3))
            rebuild(th)
            last_ranges = r.copy()
            return True

        theta, ll, beta = _fisher_scoring(
            theta, objective, max_iter=self.max_iter, tol=self.tol,
            verbose=self.verbose, reorder_hook=reorder_hook)

        theta = self._clip_theta(np.array(theta), d, estimate_nu)
        if fixed_nug:
            theta[-1] = math.log(self.nugget)
        self.theta_ = theta
        self.beta_ = beta
        self.loglik_ = ll
        self._nu_fit = math.exp(theta[1 + d]) if estimate_nu else self.nu

        # ---- variance correction (Sec. 3.4) --------------------------------
        self.b_ = 1.0
        if self.var_correction and n >= 200:
            self.b_ = self._estimate_b(rng)
        return self

    # ---------------- prediction -------------------------------------------
    def _predict_scaled(self, Xs_std, X_tr, y_tr_resid, m, noise_free=True):
        """Marginal Vecchia predictions conditioning on the m nearest training
        inputs (the 'independent' predictor).  Exact mean and variance, fully
        vectorised.  Inputs already standardised, responses already centred.
        """
        d = X_tr.shape[1]
        ranges = self.ranges_
        var = self.variance_
        nug = self.nugget_
        m = min(m, X_tr.shape[0])

        tree = cKDTree(X_tr / ranges)
        _, nb = tree.query(Xs_std / ranges, k=m)
        nb = np.atleast_2d(nb).reshape(Xs_std.shape[0], m)

        K = m + 1
        mean = np.empty(Xs_std.shape[0])
        varn = np.empty(Xs_std.shape[0])
        for a0, a1 in _chunks(Xs_std.shape[0], K, 1):
            sl = slice(a0, a1)
            Xb = np.concatenate([X_tr[nb[sl]], Xs_std[sl][:, None, :]], axis=1)
            mask = np.ones((a1 - a0, K), dtype=float)
            if noise_free:
                mask[:, K - 1] = 0.0          # predict the latent surface
            Sig, _ = _block_cov(Xb, ranges, var, nug, self.nu_,
                                 derivs=False, nug_mask=mask)
            L = _batch_chol(Sig)
            E = np.zeros((a1 - a0, K, 1))
            E[:, K - 1, 0] = 1.0
            av = _backsolve_LT(L, E)[:, :, 0]
            # a = (-b/sqrt(dv), 1/sqrt(dv))
            dv = 1.0 / av[:, K - 1] ** 2
            mean[sl] = -np.einsum("bk,bk->b", av[:, :m], y_tr_resid[nb[sl]]) \
                       / av[:, K - 1]
            varn[sl] = dv
        return mean, varn

    def predict(self, Xstar, return_std=False, return_var=False, m=None):
        """Marginal predictive mean (and sd/variance) at new inputs.

        Each test point conditions on its `m_pred` nearest training points in
        the scaled space.  Cheap, embarrassingly parallel, and gives exact
        marginal moments of the corresponding Vecchia approximation.
        """
        self._check_fitted()
        Xstar = np.ascontiguousarray(np.atleast_2d(Xstar), dtype=float)
        Xs = (Xstar - self._lo) / self._span
        m = self.m_pred if m is None else m

        Z_tr = self._design(self.X_)
        resid = self.y_ - (Z_tr @ self.beta_ if self.beta_.size else 0.0)
        mu, va = self._predict_scaled(Xs, self.X_, resid, m)
        Zs = self._design(Xs)
        if self.beta_.size:
            mu = mu + Zs @ self.beta_

        mean = self._ymu + self._ysd * mu
        if not (return_std or return_var):
            return mean
        var = (self._ysd ** 2) * self.b_ * va
        if return_var:
            return mean, var
        return mean, np.sqrt(var)

    # ---------------- joint prediction (Sec. 3.3) ---------------------------
    def _joint_factor(self, Xs, m):
        """Sparse inverse-Cholesky columns for the prediction block.

        Uses the maximin ordering of the combined scaled inputs with all
        observations ordered first.  Returns (U_op, U_pp, perm) where perm maps
        rows of Xs to their position in the prediction ordering.
        """
        n, d = self.X_.shape
        ns = Xs.shape[0]
        ranges = self.ranges_
        var, nug = self.variance_, self.nugget_

        Xall = np.vstack([self.X_, Xs])
        Xsc = Xall / ranges
        order_obs = maximin_order(Xsc[:n])
        order_pred = maximin_order(Xsc[n:]) + n
        order = np.concatenate([order_obs, order_pred])       # observations first
        Xo_sc = Xsc[order]

        nn = find_ordered_nn(Xo_sc, min(m, n), start=n)       # only pred rows needed
        Xo = Xall[order]

        rows_i, cols_j, vals = [], [], []
        K = min(m, n) + 1
        pred_rows = np.arange(n, n + ns)
        for a0, a1 in _chunks(ns, K, 1):
            sl = slice(a0, a1)
            r = pred_rows[sl]
            block = np.concatenate([nn[r, 1:K], r[:, None]], axis=1)
            mask = np.ones((a1 - a0, K))
            mask[:, K - 1] = 0.0                              # noise-free target
            # neighbours that are prediction points are also noise-free
            mask[:, :K - 1] = (block[:, :K - 1] < n).astype(float)
            Sig, _ = _block_cov(Xo[block], ranges, var, nug, self.nu_,
                                 derivs=False, nug_mask=mask)
            L = _batch_chol(Sig)
            E = np.zeros((a1 - a0, K, 1))
            E[:, K - 1, 0] = 1.0
            av = _backsolve_LT(L, E)[:, :, 0]                 # = column of U
            rows_i.append(block.ravel())
            cols_j.append(np.repeat(np.arange(a0, a1), K))
            vals.append(av.ravel())

        rows_i = np.concatenate(rows_i)
        cols_j = np.concatenate(cols_j)
        vals = np.concatenate(vals)
        Ufull = sparse.coo_matrix((vals, (rows_i, cols_j)),
                                   shape=(n + ns, ns)).tocsc()
        U_op = Ufull[:n, :]
        U_pp = Ufull[n:, :]                                   # upper triangular
        # rows of U_op are indexed by position in the *observation ordering*
        return U_op, U_pp, order[n:] - n, order_obs

    def predict_joint(self, Xstar, m=None, n_sim=0, exact_var=False, return_var=True,
                       random_state=None):
        """Joint Vecchia predictive distribution (Sec. 3.3).

        Returns a dict with 'mean', optionally 'var' and 'samples' (n_sim, n*).
        The joint law is N(mean, (U_pp U_pp')^{-1}) in the internal ordering.
        """
        self._check_fitted()
        Xstar = np.ascontiguousarray(np.atleast_2d(Xstar), dtype=float)
        Xs = (Xstar - self._lo) / self._span
        m = self.m_pred if m is None else m
        ns = Xs.shape[0]

        U_op, U_pp, perm, order_obs = self._joint_factor(Xs, m)
        inv = np.empty(ns, dtype=np.int64)
        inv[perm] = np.arange(ns)                    # ordering -> original rows

        Z_tr = self._design(self.X_)
        resid = self.y_ - (Z_tr @ self.beta_ if self.beta_.size else 0.0)

        Lt = U_pp.T.tocsr()                          # lower triangular
        rhs = -(U_op.T @ resid[order_obs])
        mu_ord = spsolve_triangular(Lt, rhs, lower=True)
        mu = mu_ord[inv]
        Zs = self._design(Xs)
        if self.beta_.size:
            mu = mu + Zs @ self.beta_
        out = {"mean": self._ymu + self._ysd * mu}

        rng = np.random.default_rng(self.random_state if random_state is None
                                     else random_state)
        if exact_var:
            S = spsolve_triangular(Lt, np.eye(ns), lower=True)   # U_pp^{-T}
            v = (S ** 2).sum(1)[inv]
            out["var"] = (self._ysd ** 2) * self.b_ * v
        if n_sim > 0:
            eps = rng.standard_normal((ns, n_sim))
            dr = spsolve_triangular(Lt, eps, lower=True)[inv]
            samples = (mu[:, None] + math.sqrt(self.b_) * dr).T
            out["samples"] = self._ymu + self._ysd * samples
            if return_var and "var" not in out:
                out["var"] = out["samples"].var(0, ddof=1)
        return out

    def sample_joint(self, Xstar, n_sim=100, m=None, random_state=None):
        """Draw joint sample paths from the predictive distribution."""
        return self.predict_joint(
            Xstar,
            m=m,
            n_sim=n_sim,
            random_state=random_state,
            return_var=False,
        )["samples"]

    # ---------------- variance correction ----------------------------------
    def _estimate_b(self, rng):
        """Sec. 3.4: pick b minimising the negative log score on an inner split."""
        n = self.X_.shape[0]
        n_in = min(2000, n // 5)
        idx = rng.permutation(n)
        te, tr = idx[:n_in], idx[n_in:]
        Z_tr = self._design(self.X_[tr])
        resid = self.y_[tr] - (Z_tr @ self.beta_ if self.beta_.size else 0.0)
        mu, va = self._predict_scaled(self.X_[te], self.X_[tr], resid,
                                       min(self.m_pred, len(tr)))
        if self.beta_.size:
            mu = mu + self._design(self.X_[te]) @ self.beta_
        va = np.maximum(va, 1e-300)
        # closed-form minimiser of sum_i [ log(b*va_i) + (y-mu)^2/(b*va_i) ] / 2
        b = float(np.mean((self.y_[te] - mu) ** 2 / va))
        return max(b, 1e-8)

    def _check_fitted(self):
        if self.theta_ is None:
            raise RuntimeError("call .fit(X, y) first")

    def summary(self):
        self._check_fitted()
        nu_label = "estimated" if self._estimate_nu else "fixed"
        s = [f"ScaledVecchiaGP(nu={'estimate' if self.nu is None else self.nu}, "
             f"m_est={self.m_est}, m_pred={self.m_pred}, trend='{self.trend}')",
             f"  Vecchia loglik (standardised y) : {self.loglik_:.3f}",
             f"  sigma^2 : {self.variance_:.6g}",
             f"  nu      : {self.nu_:.4g}  ({nu_label})",
             f"  nugget  : {self.nugget_:.6g}  (relative)",
             f"  b       : {self.b_:.4g}  (variance correction)",
             "  input relevances 1/lambda_l (unit-box scaling):"]
        for l, rel in enumerate(self.relevance_):
            s.append(f"      x[{l}] : {rel:10.4f}")
        return "\n".join(s)
