"""
Bayesian Minimisation (n-dimensional) using only NumPy and SciPy.

Algorithm:
    1. Seed with Sobol samples.
    2. Fit a Gaussian Process surrogate (Matérn 5/2, ARD length-scales,
       locally-periodic Matérn for periodic dims, linear mean function
       on non-periodic dimensions only, mean/std normalisation).
    3. Maximise Expected Improvement via batched Sobol sampling (global)
       followed by L-BFGS-B with analytic gradients (local polish).
    4. Evaluate the true objective at the chosen point and repeat.

Copyright (c) 2026 Gonzalo Morras (gonzalo.morras@aei.mpg.de)

This file is part of pybop.
Licensed under the Apache-2.0 License. See the LICENSE file in the project root for details.
"""

import numpy as np
from scipy.linalg import solve_triangular, cho_solve
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.special import erfc
import warnings

# ---------------------------------------------------------------------------
# Helper to generate quasi-random Sobol samples
# ---------------------------------------------------------------------------

def _round_up_to_power_of_two(n: int, name: str = "n", stacklevel: int = 2) -> int:
    """
    Return the smallest power of two that is >= n.

    If n is already a power of two it is returned unchanged.  Otherwise a
    UserWarning is raised and the rounded-up value is returned.

    Parameters
    ----------
    n : int
        Value to check. Must be a positive integer.
    name : str, optional
        Variable name shown in the warning message. Defaults to ``"n"``.
    stacklevel : int, optional
        Passed to ``warnings.warn`` to point the warning at the caller.
        Defaults to ``2``.

    Returns
    -------
    int
        The smallest power of two >= n.

    Raises
    ------
    ValueError
        If n is not a positive integer.
    """
    if n < 1:
        raise ValueError(f"{name} must be a positive integer; got {n}.")

    log2_n = np.log2(n)
    ceil_log2_n = int(np.ceil(log2_n))

    if log2_n == ceil_log2_n:
        return n

    next_power = 2 ** ceil_log2_n
    warnings.warn(
        f"{name}={n} is not a power of two. Rounding up to {next_power}.",
        UserWarning,
        stacklevel=stacklevel,
    )
    return next_power

def _generate_Sobol_samples(
    n_samples: int,
    bounds: list | np.ndarray,
    scramble: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate quasi-random samples using a Sobol sequence, scaled to the
    provided parameter bounds.

    Sobol sequences are low-discrepancy sequences that cover the parameter
    space more uniformly than pseudo-random sampling, making them well-suited
    for numerical integration and global optimisation tasks.

    .. note::
        Sobol sequences require the number of samples to be a power of two.
        If ``n_samples`` is not a power of two, it is rounded up to the next
        power of two and a ``UserWarning`` is raised.

    .. note::
        ``seed`` has no effect when ``scramble=False``, as randomisation is
        only applied during scrambling.

    Parameters
    ----------
    n_samples : int
        Desired number of samples. Must be a positive integer. If not a power
        of two, it will be rounded up to the nearest power of two.
    bounds : array-like of shape (n_dims, 2)
        Array of lower and upper bounds for each dimension, where
        ``bounds[i, 0]`` and ``bounds[i, 1]`` are the lower and upper
        bounds of the i-th dimension, respectively.
    scramble : bool, optional
        If ``True`` (default), the Sobol sequence is scrambled using Owen
        scrambling, which improves uniformity and allows reproducibility via
        ``seed``. If ``False``, the deterministic, unscrambled sequence is
        used.
    seed : int or None, optional
        Random seed for reproducibility. Only used when ``scramble=True``.
        Defaults to ``None``.

    Returns
    -------
    np.ndarray of shape (n_samples, n_dims)
        Quasi-random samples scaled to the provided bounds.

    Warns
    -----
    UserWarning
        If ``n_samples`` is not a power of two, a warning is raised indicating
        the value has been rounded up.

    """
    
    n_samples  = _round_up_to_power_of_two(n_samples, name="n_samples", stacklevel=3)
    log2_n     = int(np.log2(n_samples))
    bounds_arr = np.asarray(bounds)
    sobol      = qmc.Sobol(len(bounds_arr), scramble=scramble, seed=seed)
    X_sobol    = sobol.random_base2(log2_n)
    return qmc.scale(X_sobol, bounds_arr[:, 0], bounds_arr[:, 1])

# ---------------------------------------------------------------------------
# Distance helper: supports both standard and periodic dimensions
# ---------------------------------------------------------------------------

def _compute_r2(X1: np.ndarray, X2: np.ndarray,
                length_scale: np.ndarray,
                periodic_dims: list[int] | None = None,
                periods_norm:  np.ndarray | None = None) -> np.ndarray:
    """
    Pairwise squared (locally-periodic ARD) Matérn distance r2[i, j] between
    rows of X1 (n1, d) and X2 (n2, d), in normalised space.

    The squared distance is accumulated one dimension at a time, forming the
    pairwise differences *before* squaring:

        r2[i, j] = Σ_k ( δ_k(x1_i, x2_j) / ls_k )²

    where δ_k is the raw difference (x1_k - x2_k) for standard dims and the
    locally-periodic difference 2·sin(π(x1_k - x2_k)/T_k) for periodic dims.
    Differencing before squaring keeps the result numerically stable for
    nearby points.  Only one (n1, n2) buffer is held per dimension, so no
    (n1, n2, d) difference tensor is materialised.

    Parameters
    ----------
    X1, X2         : (n1, d), (n2, d) in normalised space
    length_scale   : (d,)
    periodic_dims  : list of dimension indices that are periodic
    periods_norm   : (d,) effective period in normalised space
                     (only entries at periodic_dims are used)

    Returns
    -------
    r2 : (n1, n2) total squared distance (≥ 0 by construction).
    """
    periodic_dims = periodic_dims or []
    per_set       = set(periodic_dims)

    r2 = np.zeros((X1.shape[0], X2.shape[0]))
    for k in range(X1.shape[1]):
        if k in per_set:
            # δ_k = 2·sin(π(x1-x2)/T)/ls.  Expand via the angle-difference identity
            # sin(α-β)=sinα·cosβ-cosα·sinβ so the transcendental functions are evaluated on the
            # 1-D coordinates (2*(n1+n2) calls) rather than on the 2-D difference matrix (n1*n2 calls).
            f  = np.pi / periods_norm[k]
            a1 = f * X1[:, k]
            a2 = f * X2[:, k]
            w  = 2.0 / length_scale[k]
            delta = (w * np.sin(a1))[:, None] * np.cos(a2)[None, :] - (w * np.cos(a1))[:, None] * np.sin(a2)[None, :]
        else:
            w  = 1.0 / length_scale[k]
            delta = (w * X1[:, k])[:, None] - (w * X2[:, k])[None, :]
        r2 += delta * delta
    return r2


# ---------------------------------------------------------------------------
# Matérn 5/2 kernel  (ARD, with optional periodic dimensions)
# ---------------------------------------------------------------------------

_SQRT5 = float(np.sqrt(5.0))


def _matern52(X1: np.ndarray, X2: np.ndarray,
              length_scale: np.ndarray, variance: float,
              periodic_dims: list[int] | None = None,
              periods_norm:  np.ndarray | None = None) -> np.ndarray:
    """
    Matérn 5/2 covariance matrix between rows of X1 (n1, d) and X2 (n2, d).

    For periodic dimensions the Euclidean difference |x1_k - x2_k| is
    replaced by  2|sin(π·Δ/T_k)|  — the standard 'locally periodic Matérn'
    construction, which preserves positive-definiteness.  The squared
    distance is delegated to :func:`_compute_r2`.

    length_scale : (d,)  — one scale per dimension (ARD)
    periodic_dims: list of int — indices of periodic dimensions
    periods_norm : (d,)  — period per dim in normalised space
                           (only entries at periodic_dims are used)
    """
    r2      = _compute_r2(X1, X2, length_scale, periodic_dims, periods_norm)
    r       = np.sqrt(r2)
    sqrt5_r = _SQRT5 * r
    return variance * (1.0 + sqrt5_r + (5.0 / 3.0) * r2) * np.exp(-sqrt5_r)


# ---------------------------------------------------------------------------
# Gaussian Process
# ---------------------------------------------------------------------------

class GaussianProcess:
    """
    GP regressor with:
      - Matérn 5/2 kernel with ARD length-scales
      - Optional locally-periodic Matérn for specified dimensions
      - Linear mean function  m(x) = β₀ + β·x  fitted by least squares
        on normalised data, using ONLY the non-periodic dimensions as
        regressors (periodic dims are excluded from the linear trend).
      - Mean/std normalisation for both X and y
      - Screened hyperparameter restarts

    Parameters
    ----------
    periodic_dims : list[int] | None
        Indices of dimensions that are periodic.
    periods : list[float] | np.ndarray | None
        Period for each dimension listed in periodic_dims, in the
        *original* (un-normalised) space.  Must have the same length
        as periodic_dims.
    length_scale_bounds : (float, float)
    variance_bounds     : (float, float)
    noise_variance_bounds : (float, float)
    normalize : bool
    deterministic : bool
    jitter : float
    n_restarts : int
    n_screen : int
        Number of Sobol candidates used to screen hyperparameter starting
        points. Should be a power of two; if not, it will be rounded up
        and a UserWarning will be raised.
    random_state : int | None
    """

    def __init__(
        self,
        periodic_dims: list[int] | None = None,
        periods:       list[float] | np.ndarray | None = None,
        length_scale_bounds:   tuple = (1e-4, 100.0),
        variance_bounds:       tuple = (1e-4, 10.0),
        noise_variance_bounds: tuple = (1e-5, 1.0),
        normalize:     bool  = True,
        deterministic: bool  = True,
        jitter:        float = 1e-10,
        n_restarts:    int   = 10,
        n_screen:      int   = 128,
        random_state:  int | None = None,
    ):
        self.periodic_dims = list(periodic_dims) if periodic_dims is not None else []
        self.periods       = (np.asarray(periods, dtype=float)
                              if periods is not None else np.array([]))

        if len(self.periodic_dims) != len(self.periods):
            raise ValueError(
                "periodic_dims and periods must have the same length; "
                f"got {len(self.periodic_dims)} and {len(self.periods)}."
            )

        self.length_scale_bounds   = length_scale_bounds
        self.variance_bounds       = variance_bounds
        self.noise_variance_bounds = noise_variance_bounds
        self.normalize     = normalize
        self.deterministic = deterministic
        self.jitter        = jitter
        self.n_restarts    = n_restarts
        self.n_screen      = _round_up_to_power_of_two(n_screen, name="n_screen", stacklevel=2)
        self.rng           = np.random.RandomState(random_state)

        self.length_scale_:  np.ndarray | None = None
        self.variance_       = 1.0
        self.noise_variance_ = 0.0 if deterministic else 1e-3

        # Linear mean function: β[0] = intercept, β[1:] = slopes for
        # non-periodic dims (in normalised space).
        self._beta: np.ndarray | None = None
        # Which column indices in normalised X are used as regressors.
        self._mean_dims: list[int] = []

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self._pd_train_sq: np.ndarray | None = None
        self._L:     np.ndarray | None = None
        self._alpha: np.ndarray | None = None

        self.X_mean_: np.ndarray | None = None
        self.X_std_:  np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_:  float = 1.0

        # Effective periods in normalised space (filled in fit())
        self._periods_norm: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_X(self, X: np.ndarray) -> np.ndarray:
        if self.normalize:
            return (X - self.X_mean_) / self.X_std_
        return X

    def _normalise_y(self, y: np.ndarray) -> np.ndarray:
        if self.normalize:
            return (y - self.y_mean_) / self.y_std_
        return y

    def _mean(self, Xn: np.ndarray) -> np.ndarray:
        """
        Evaluate the linear mean function at normalised Xn.
        Only non-periodic dimensions enter as regressors.
        Shape: (n,) if Xn is (n, d), or scalar if Xn is (d,).
        """
        if Xn.ndim == 1:
            return float(self._beta[0] + self._beta[1:] @ Xn[self._mean_dims])
        return Xn[:, self._mean_dims] @ self._beta[1:] + self._beta[0]

    def _dmean_dxn(self) -> np.ndarray:
        """
        Gradient of the mean function w.r.t. normalised x.
        Returns (d,) array — zero for periodic dims, β[1+j] for non-periodic.
        """
        d  = self.X_train_.shape[1]
        dm = np.zeros(d)
        for j, k in enumerate(self._mean_dims):
            dm[k] = self._beta[1 + j]
        return dm

    # ------------------------------------------------------------------
    # Kernel helpers (with periodic support)
    # ------------------------------------------------------------------

    def _kernel_matrix(self, X1, X2):
        return _matern52(X1, X2,
                         self.length_scale_, self.variance_,
                         self.periodic_dims, self._periods_norm)

    # ------------------------------------------------------------------
    # Log-marginal likelihood with analytic gradients
    # ------------------------------------------------------------------

    def _log_marginal_likelihood(self, log_params: np.ndarray,
                                 compute_grad: bool = True):
        """
        Negative log-marginal likelihood and its analytic gradient w.r.t.
        log-hyperparameters.

        When ``compute_grad`` is False only the (scalar) negative log-marginal
        likelihood is returned and the gradient is not evaluated.  This is used
        by the hyperparameter screening phase, which discards the gradient, and
        avoids the O(n³) ``K⁻¹`` solve plus the per-dimension gradient sums.
        The returned value is identical to ``...(log_params)[0]``.

        Matérn 5/2 partial derivatives
        --------------------------------
        For a *standard* dimension k:
            d_k = diff_k / ls_k
            dr2/d(log ls_k) = -2 * r2_per_k          (chain rule in log space)

        For a *periodic* dimension k  (with p_k = 2sin(πΔ/T_k)):
            d_k = p_k / ls_k
            dr2/d(log ls_k) = -2 * r2_per_k          (same form!)

        Common factor:
            dK/d(log ls_k) = K_signal * (5/3) * (1 + √5r)/(1 + √5r + 5r²/3)
                             * r2_per_k            [times extra factor below]

        More explicitly, using K = var*(1+√5r+5r²/3)*exp(-√5r):
            dK/dr = var * exp(-√5r) * [√5 + (10/3)r - √5*(1+√5r+5r²/3)]
                  = var * exp(-√5r) * (5/3) * (1 + √5r) * ... [use r2 form]

        We use the identity:
            d(-logp)/d(log ls_k) = -½ tr[W * dK/d(log ls_k)]
            dK/d(log ls_k) = var * exp(-√5r) * (5/3) * (1+√5r) * r2_per_k * 2
        """
        d     = self.X_train_.shape[1]
        ls    = np.exp(log_params[:d])
        var   = np.exp(log_params[d])
        noise = 0.0 if self.deterministic else np.exp(log_params[d + 1])
        y     = self.y_train_
        n     = len(y)
        n_params = d + 1 if self.deterministic else d + 2

        def _logp(L):
            alpha_L = solve_triangular(L, y,         lower=True,  check_finite=False)
            alpha   = solve_triangular(L.T, alpha_L, lower=False, check_finite=False)
            log_lik = (
                -0.5 * y @ alpha
                - np.sum(np.log(np.diag(L)))
                - 0.5 * n * np.log(2.0 * np.pi)
            )
            return alpha, log_lik

        # Squared distances from the pre-computed SQUARED effective differences
        # (hyperparameter-independent; built once per fit).  r2 is a single
        # contraction pd_sq · ls⁻², avoiding a per-call (pd/ls)² over (n,n,d).
        # Fall back to computing pd_sq if used without a prior fit.
        pd_sq = self._pd_train_sq
        if pd_sq is None:
            pd_sq = self._effective_differences(self.X_train_) ** 2
        inv_ls2  = 1.0 / ls ** 2                          # (d,)
        r2       = pd_sq @ inv_ls2                         # (n,n,d)·(d,) -> (n,n)
        r        = np.sqrt(r2)
        exp_r    = np.exp(-_SQRT5 * r)
        K_signal = var * (1.0 + _SQRT5 * r + 5.0 / 3.0 * r2) * exp_r
        K        = K_signal.copy() if compute_grad else K_signal
        K.flat[::n + 1] += noise + self.jitter           # add to diagonal in place

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return (1e10, np.zeros(n_params)) if compute_grad else 1e10

        alpha, log_lik = _logp(L)

        # ---- Value-only path (hyperparameter screening) ----
        if not compute_grad:
            return -log_lik

        K_inv = cho_solve((L, True), np.eye(n), check_finite=False)
        W     = np.outer(alpha, alpha) - K_inv

        grad = np.empty(n_params)

        # dK/d(log ls_k) = var·exp(-√5r)·(5/3)·(1+√5r)·(pd_sq_k/ls_k²)  (log-space
        # chain rule).  The 1/ls_k² factor is pulled out of the einsum so the
        # hyperparameter-independent pd_sq is contracted against W directly.
        WC = W * (var * exp_r * (5.0 / 3.0) * (1.0 + _SQRT5 * r))
        grad[:d] = -0.5 * inv_ls2 * np.einsum("ij,ijk->k", WC, pd_sq)
        grad[d]  = -0.5 * np.einsum("ij,ij->", W, K_signal)
        if not self.deterministic:
            grad[d + 1] = -0.5 * noise * np.trace(W)

        return -log_lik, grad

    def _optimise_hyperparams(self) -> None:
        d = self.X_train_.shape[1]
        log_bounds = (
            [(np.log(self.length_scale_bounds[0]),
              np.log(self.length_scale_bounds[1]))] * d
            + [(np.log(self.variance_bounds[0]),
                np.log(self.variance_bounds[1]))]
        )
        if not self.deterministic:
            log_bounds += [(np.log(self.noise_variance_bounds[0]),
                            np.log(self.noise_variance_bounds[1]))]

        # Phase 1 — screen
        screen_pts = _generate_Sobol_samples(
            n_samples  = self.n_screen,
            bounds     = log_bounds,
            scramble   = True,
            seed       = int(self.rng.randint(0, 2**31 - 1)),
        )
        screen_vals = np.array([
            self._log_marginal_likelihood(pt, compute_grad=False)
            for pt in screen_pts
        ])
        best_screen_idx = np.argpartition(screen_vals, self.n_restarts)[:self.n_restarts]
        best_starts     = screen_pts[best_screen_idx]

        # Phase 2 — L-BFGS-B
        best_val, best_log_params = np.inf, None
        for x0 in best_starts:
            res = minimize(
                self._log_marginal_likelihood, x0,
                bounds=log_bounds, method="L-BFGS-B",
                jac=True,
            )
            if res.fun < best_val:
                best_val, best_log_params = res.fun, res.x

        params = np.exp(best_log_params)
        self.length_scale_ = params[:d]
        self.variance_     = params[d]
        if not self.deterministic:
            self.noise_variance_ = params[d + 1]

    def _build_cholesky(self) -> None:
        X, y = self.X_train_, self.y_train_
        n    = len(y)
        K    = self._kernel_matrix(X, X)
        K   += (self.noise_variance_ + self.jitter) * np.eye(n)
        self._L = np.linalg.cholesky(K)
        v = solve_triangular(self._L, y, lower=True, check_finite=False)
        self._alpha = solve_triangular(self._L.T, v, lower=False, check_finite=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """
        Fit the GP to training data.

        Steps:
          1. Normalise X and y.
          2. Compute effective periods in normalised space.
          3. Fit a linear mean function using ONLY non-periodic dimensions.
          4. Optimise hyperparameters.
          5. Cache Cholesky.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        d = X.shape[1]

        # ---- Normalisation ----
        if self.normalize:
            self.X_mean_ = np.mean(X, axis=0)
            self.X_std_  = np.std(X,  axis=0) + 1e-12
            self.y_mean_ = float(np.mean(y))
            self.y_std_  = float(np.std(y)  + 1e-12)
        else:
            self.X_mean_ = np.zeros(d)
            self.X_std_  = np.ones(d)
            self.y_mean_ = 0.0
            self.y_std_  = 1.0

        Xn = self._normalise_X(X)
        yn = self._normalise_y(y)

        # ---- Effective periods in normalised space ----
        # period_norm_k = period_k / X_std_k
        self._periods_norm = np.ones(d)   # only used for periodic dims
        for i, k in enumerate(self.periodic_dims):
            self._periods_norm[k] = self.periods[i] / self.X_std_[k]

        # ---- Non-periodic dims used for linear mean ----
        self._mean_dims = [k for k in range(d) if k not in self.periodic_dims]

        # ---- Fit linear mean on non-periodic dims only ----
        n_mean_features = len(self._mean_dims)
        use_linear_mean = (
            n_mean_features > 0
            and Xn.shape[0] > 2 * (n_mean_features + 1)
        )

        if use_linear_mean:
            Phi     = np.column_stack(
                [np.ones(len(Xn)), Xn[:, self._mean_dims]]
            )                                                     # (n, 1+|mean_dims|)
            PhiTPhi = Phi.T @ Phi + 1e-6 * np.eye(Phi.shape[1])
            self._beta = np.linalg.solve(PhiTPhi, Phi.T @ yn)    # (1+|mean_dims|,)
        else:
            self._beta    = np.zeros(n_mean_features + 1)
            self._beta[0] = np.mean(yn)

        residuals = yn - self._mean(Xn)

        self.X_train_ = Xn
        self.y_train_ = residuals

        # Per-dimension SQUARED effective differences of the training set.
        # These depend only on the (fixed) inputs and periods, not on the
        # hyperparameters, so squaring them once here lets every likelihood
        # evaluation get r2 from a single contraction  r2 = pd_sq · ls⁻²
        # instead of rebuilding (pd/ls)² over the (n, n, d) tensor each call.
        self._pd_train_sq = self._effective_differences(Xn) ** 2

        if self.length_scale_ is None:
            self.length_scale_ = np.ones(d)

        self._optimise_hyperparams()
        self._build_cholesky()
        return self

    def _effective_differences(self, X: np.ndarray) -> np.ndarray:
        """
        Per-dimension differences between all training pairs, shape (n, n, d):
        the raw difference x_i - x_j for standard dims, and the locally-periodic
        2·sin(π(x_i - x_j)/T_k) for periodic dims.  Dividing by the length
        scale and squaring then yields the per-dim squared distance.
        """
        pd = X[:, None, :] - X[None, :, :]                       # (n, n, d)
        for k in self.periodic_dims:
            pd[:, :, k] = 2.0 * np.sin(np.pi * pd[:, :, k] / self._periods_norm[k])
        return pd

    def predict(self, X: np.ndarray,
                return_std: bool = False) -> np.ndarray | tuple:
        """
        Posterior mean and (optionally) std at X, in original scale.
        """
        X  = np.asarray(X, dtype=float)
        Xn = self._normalise_X(X)

        K_s  = _matern52(self.X_train_, Xn,
                         self.length_scale_, self.variance_,
                         self.periodic_dims, self._periods_norm)
        mu_n = self._mean(Xn) + K_s.T @ self._alpha

        mu = mu_n * self.y_std_ + self.y_mean_ if self.normalize else mu_n

        if not return_std:
            return mu

        v        = solve_triangular(self._L, K_s, lower=True, check_finite=False)
        var_post = self.variance_ - np.sum(v ** 2, axis=0)
        sigma    = np.sqrt(np.maximum(var_post, 0.0))
        if self.normalize:
            sigma = sigma * self.y_std_

        return mu, sigma

    def predict_with_gradients(self, x: np.ndarray) -> tuple:
        """
        Posterior mean, std, and their analytic gradients at a single point x.

        For non-periodic dim k:
            d(pd_k)/d(xn_k) = 1  →  same as before

        For periodic dim k  (pd_k = 2 sin(π Δ_k / T_k),  Δ_k = xn_k - x_tr_k):
            d(pd_k)/d(xn_k) = 2π/T_k · cos(π Δ_k / T_k)

        Combined kernel gradient:
            dk/d(xn_k) = (5/3) · var · exp(-√5 r) · (1 + √5 r)
                         · (pd_k / ls_k²) · d(pd_k)/d(xn_k)
        """
        x  = np.asarray(x, dtype=float)
        xn = self._normalise_X(x)

        X_tr    = self.X_train_
        ls, var = self.length_scale_, self.variance_
        n_tr, d = X_tr.shape

        diff = X_tr - xn[None, :]                              # (n_train, d)

        # Per-dim "effective difference" pd_k: equal to the raw difference for
        # standard dims, and 2·sin(π Δ/T) for periodic dims.
        pd = diff.copy()                                      # (n_train, d)
        for k in self.periodic_dims:
            T_k      = self._periods_norm[k]
            pd[:, k] = 2.0 * np.sin(np.pi * diff[:, k] / T_k)

        r2_per   = (pd / ls) ** 2                              # (n_train, d)
        r2       = r2_per.sum(axis=-1)                         # (n_train,)
        r        = np.sqrt(np.maximum(r2, 0.0))
        exp_term = np.exp(-np.sqrt(5.0) * r)
        k_s      = var * (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * r2) * exp_term

        mu_n = float(self._mean(xn) + k_s @ self._alpha)

        # ---- Gradient of k_s w.r.t. xn ----
        # common scalar factor per training point
        factor = (5.0 / 3.0) * var * exp_term * (1.0 + np.sqrt(5.0) * r)
        # d(pd_k)/d(xn_k): 1 for standard dims, 2π/T·cos(π Δ/T) for periodic.
        dpd_dxn = np.ones((n_tr, d))
        for k in self.periodic_dims:
            T_k           = self._periods_norm[k]
            dpd_dxn[:, k] = (2.0 * np.pi / T_k) * np.cos(np.pi * diff[:, k] / T_k)

        # (n_train, d):  factor_i · (pd_ik / ls_k²) · d(pd_ik)/d(xn_k)
        dk_dxn = factor[:, None] * (pd / ls ** 2) * dpd_dxn

        # Gradient of mean function w.r.t. normalised x
        dmu_dxn  = self._dmean_dxn() + dk_dxn.T @ self._alpha  # (d,)

        # Posterior std — one triangular solve for k_s and dk_dxn stacked.
        sol      = solve_triangular(self._L, np.column_stack((k_s, dk_dxn)),
                                    lower=True, check_finite=False)   # (n_train, 1+d)
        v        = sol[:, 0]
        dv_dxn   = sol[:, 1:]                                       # (n_train, d)
        var_post = max(float(var - v @ v), 1e-18)
        sigma_n  = np.sqrt(var_post)

        dvar_dxn   = -2.0 * dv_dxn.T @ v                            # (d,)
        dsigma_dxn = dvar_dxn / (2.0 * sigma_n)

        # ---- Rescale to original space ----
        if self.normalize:
            # d/dx_orig = d/dx_norm * (1/X_std)
            dmu_dx    = dmu_dxn    * self.y_std_ / self.X_std_
            dsigma_dx = dsigma_dxn * self.y_std_ / self.X_std_
            mu    = mu_n    * self.y_std_ + self.y_mean_
            sigma = sigma_n * self.y_std_
        else:
            mu, sigma = mu_n, sigma_n
            dmu_dx, dsigma_dx = dmu_dxn, dsigma_dxn

        return float(mu), float(sigma), dmu_dx, dsigma_dx


# ---------------------------------------------------------------------------
# Acquisition function — Negative Log Expected Improvement (minimisation)
# ---------------------------------------------------------------------------

def neg_logEI_z_small(z: np.ndarray, compute_derivatives: bool=False):
    """
    Compute -Log(EI) = nlogEI_z = -log(z Φ(z) + φ(z)) when z->-\infinity.
    
    We do an asymptotic expansion at z->-\infinity with a (10,10) Pade resummation for better convergence.
    
    These approximation are accurate to numerical precission for z<-10.
    
    If compute_derivatives is True, return also
        -(1/sigma)\partial\log(EI)/\partial\mu    =  Φ(z)/(z Φ(z) + φ(z))
        -(1/sigma)\partial\log(EI)/\partial\sigma = -φ(z)/(z Φ(z) + φ(z))
    """
    
    z2 = z*z
    PadeTerm = (10395. + z2*(17325. + z2*(6930. + z2*(990. + z2*(55. + z2)))))/(6555. + z2*(4680. + z2*(840. + z2*(52. + z2))))
    nlogEI = 0.5*z2 + np.log(PadeTerm) + 0.5*np.log(2*np.pi)
    if not compute_derivatives:
        return nlogEI
    else:
        dmu_term = -(3840. + z2*(12645. + z2*(6090. + z2*(938. + z2*(54. + z2)))))/((6555. + z2*(4680. + z2*(840. + z2*(52. + z2))))*z)
        dsigma_term = -PadeTerm
        return nlogEI, dmu_term, dsigma_term

def neg_logEI_z_large(z: np.ndarray, compute_derivatives: bool=False):
    """
    Compute -Log(EI) = nlogEI_z = -log(z Φ(z) + φ(z)) using the definition directly.
    
    If compute_derivatives is True, return also
        -(1/sigma)\partial\log(EI)/\partial\mu    =  Φ(z)/(z Φ(z) + φ(z))
        -(1/sigma)\partial\log(EI)/\partial\sigma = -φ(z)/(z Φ(z) + φ(z))
    """

    
    cdf_z = 0.5*erfc(-z/np.sqrt(2))
    pdf_z = np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
    EI = z * cdf_z + pdf_z
    nlogEI = -np.log(EI)
    if not compute_derivatives:
        return nlogEI
    else:
        dmu_term    =  cdf_z/EI
        dsigma_term = -pdf_z/EI
        return nlogEI, dmu_term, dsigma_term

def neg_logEI(X: np.ndarray, gp: GaussianProcess,
              y_best: float, xi: float = 0.01, z_small: float=-10.) -> np.ndarray:
    """
    Vectorised negative Log Expected Improvement for minimisation.

    neg_logEI = -log[σ·(z·Φ(z) + φ(z))],   z = (y_best - μ - ξ) / σ
    
    z_small represents the values of z below which we use asymptotic expansions of neg_logEI to avoid numerical errors.
    """
    mu, sigma = gp.predict(X, return_std=True)
    nlogei = np.full(mu.shape, np.inf)
    mask_sigma = sigma > 1e-14*gp.y_std_
    s = sigma[mask_sigma]
    m = y_best - mu[mask_sigma] - xi
    z = m/s

    # Values of z to which we apply small z approximation
    nlogei_z = np.zeros_like(z)
    mask_zsmall = z<z_small
    if np.any(mask_zsmall):
        nlogei_z[mask_zsmall] = neg_logEI_z_small(z[mask_zsmall], compute_derivatives=False)
    # Values of z to which we apply exact formula
    mask_zlarge = np.logical_not(mask_zsmall)
    if np.any(mask_zlarge):
        nlogei_z[mask_zlarge] = neg_logEI_z_large(z[mask_zlarge], compute_derivatives=False)

    nlogei[mask_sigma] = nlogei_z - np.log(s)
    return nlogei

def neg_logEI_with_gradient(x: np.ndarray, gp: GaussianProcess,
                            y_best: float, xi: float = 0.01, z_small: float=-10.) -> tuple:
    """
    Negated log(EI) and its analytic gradient at a single point, for L-BFGS-B.

        -dlog(EI)/dx = (dμ/dx · Φ(z) - dσ/dx · φ(z))/EI

    z_small represents the values of z below which we use asymptotic expansions of neg_logEI to avoid numerical errors.
    """
    mu, sigma, dmu_dx, dsigma_dx = gp.predict_with_gradients(x)

    if sigma < 1e-14*gp.y_std_:
        return np.inf, np.zeros_like(dmu_dx)

    z = (y_best - mu - xi) / sigma
    
    if z<z_small:
        nlogEI_z, dmu_term, dsigma_term = neg_logEI_z_small(z, compute_derivatives=True)
    else:
        nlogEI_z, dmu_term, dsigma_term = neg_logEI_z_large(z, compute_derivatives=True)
    
    nlogEI  = nlogEI_z - np.log(sigma)
    grad_nlogEI = (dmu_dx*dmu_term + dsigma_dx*dsigma_term)/sigma

    return nlogEI, grad_nlogEI

# ---------------------------------------------------------------------------
# Bayesian Minimisation loop
# ---------------------------------------------------------------------------

def bayesian_minimization(
    objective,
    bounds: list[tuple[float, float]],
    args:          tuple = (),
    kwargs:        dict | None = None,
    n_initial:     int   = 32,
    n_calls:       int   = 100,
    xi0:           float = 0.1,
    xif:           float = 1e-4,
    n_samples:     int   = 131_072,
    batch_size:    int   = 512,
    n_polish:      int   = 50,
    gp_kwargs:     dict | None = None,
    periodic_dims: list[int] | None = None,
    periods:       list[float] | np.ndarray | None = None,
    random_state:  int  | None = None,
    verbose:       bool = True,
    deterministic: bool = True,
    final_polish:  bool = True,
) -> dict:
    """
    Minimise *objective* using Bayesian optimisation.

    Parameters
    ----------
    objective      : f(x: ndarray (d,), *args, **kwargs) -> float
    bounds         : list of (lower, upper) per dimension
    args           : extra positional arguments passed to objective
    kwargs         : extra keyword arguments passed to objective
    n_initial      : Sobol seed points (rounded up to next power of two)
    n_calls        : total number of objective evaluations
    xi0            : initial EI exploration parameter
    xif            : final EI exploration parameter (xif < xi0)
    n_samples      : Sobol candidates per BO iteration (rounded up to next power of two)
    batch_size     : EI evaluation batch size
    n_polish       : top candidates handed to L-BFGS-B
    gp_kwargs      : extra kwargs forwarded to GaussianProcess()
    periodic_dims  : list of dimension indices that are periodic
    periods        : period for each dim in periodic_dims
    random_state   : int seed
    verbose        : print progress
    deterministic  : treat objective as noise-free
    final_polish   : do a final GP-mean minimisation step

    Returns
    -------
    dict:
        'x_best' : (d,)                    — best point found
        'y_best' : float                   — lowest objective value
        'X'      : (n_calls, d) — all evaluated points
        'y'      : (n_calls,)   — all objective values
        'gp'     : GaussianProcess         — fitted on final dataset
    """
    kwargs = kwargs or {}

    def call_objective(x):
        return objective(x, *args, **kwargs)

    rng = np.random.RandomState(random_state)
    bounds_arr = np.asarray(bounds, dtype=float)
    d = len(bounds_arr)

    # Validate periodic arguments
    periodic_dims = list(periodic_dims) if periodic_dims is not None else []
    if periods is not None:
        periods = np.asarray(periods, dtype=float)
    else:
        periods = np.array([])

    if len(periodic_dims) != len(periods):
        raise ValueError(
            "periodic_dims and periods must have the same length."
        )

    # --- 1. Sobol Initialisation ---
    
    n_initial = _round_up_to_power_of_two(n_initial, name="n_initial", stacklevel=2)
    
    X = _generate_Sobol_samples(
        n_initial,
        bounds,
        scramble=True,
        seed=int(rng.randint(0, 2 ** 31 - 1)),
    )
    y = np.array([call_objective(x) for x in X])
    
    if verbose:
        print(f"{'Iter':>5}  {'μ':>10}  {'σ':>10}  {'-log(EI)':>10}  "
              f"{'y':>10}  {'y_best':>10}")
        for i, yi in enumerate(y):
            print(f"{i + 1:>5}  {'—':>10}  {'—':>10}  {'—':>10}  "
                  f"{yi:>10.4g}  {y[:i + 1].min():>10.4g}")

    # ---- 2. Build GP ----
    merged_gp_kwargs = {
        "deterministic": deterministic,
        "periodic_dims": periodic_dims,
        "periods":       periods,
        **(gp_kwargs or {}),
    }
    gp = GaussianProcess(
        **merged_gp_kwargs,
        random_state=int(rng.randint(0, 2 ** 31 - 1)),
    )

    # ---- 3. BO loop ----
    n_iter = n_calls - n_initial
    if final_polish:
        n_iter -= 1

    n_samples = _round_up_to_power_of_two(n_samples, name="n_samples", stacklevel=2)
    for iteration, xi_raw in enumerate(np.geomspace(xi0, xif, n_iter)):

        gp.fit(X, y)
        y_best = y.min()
        xi = xi_raw * gp.y_std_

        # Phase 1 — batched Sobol sampling
        candidates = _generate_Sobol_samples(
            n_samples,
            bounds,
            scramble=True,
            seed=int(rng.randint(0, 2 ** 31 - 1)),
        )

        nlogEIs = np.full(n_samples, np.inf)
        for i_0 in range(0, n_samples, batch_size):
            i_f = min(i_0 + batch_size, n_samples)
            nlogEIs[i_0:i_f] = neg_logEI(
                candidates[i_0:i_f], gp, y_best, xi
            )

        # Phase 2 — L-BFGS-B from top n_polish candidates
        top_idxs    = np.argpartition(nlogEIs, n_polish)[:n_polish]
        top_starts  = candidates[top_idxs]
        top_nlogEIs = nlogEIs[top_idxs]

        min_idx = np.argmin(top_nlogEIs)
        best_x, best_nlogEI = top_starts[min_idx], top_nlogEIs[min_idx]
        for x0 in top_starts:
            res = minimize(
                neg_logEI_with_gradient, x0,
                args=(gp, y_best, xi),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            if res.fun < best_nlogEI:
                best_nlogEI = res.fun
                best_x   = res.x

        x_next = np.clip(best_x, bounds_arr[:, 0], bounds_arr[:, 1])
        y_next = call_objective(x_next)

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        if verbose:
            mu_x, sigma_x = gp.predict(x_next[None, :], return_std=True)
            print(f"{n_initial + iteration + 1:>5}  "
                  f"{mu_x[0]:>10.4g}  {sigma_x[0]:>10.4g}  {best_nlogEI:>10.4g}  "
                  f"{y_next:>10.4g}  {y.min():>10.4g}")

    # ---- 4. Final GP-mean polish ----
    if final_polish:

        gp.fit(X, y)

        def gp_mean(x):
            mu, _, dmu, _ = gp.predict_with_gradients(x)
            return mu, dmu

        mu_res = minimize(
            gp_mean,
            x0=X[y.argmin()],
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        )
        x_next = np.clip(mu_res.x, bounds_arr[:, 0], bounds_arr[:, 1])
        y_next = call_objective(x_next)

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        if verbose:
            mu_x, sigma_x = gp.predict(x_next[None, :], return_std=True)
            print(f"{n_calls:>5}  "
                  f"{mu_x[0]:>10.4g}  {sigma_x[0]:>10.4g}  {'—':>10}  "
                  f"{y_next:>10.4g}  {y.min():>10.4g}")

    gp.fit(X, y)
    best_idx = y.argmin()
    return {
        "x_best": X[best_idx],
        "y_best": float(y[best_idx]),
        "X":      X,
        "y":      y,
        "gp":     gp,
    }
