"""
Tests for the periodic-dimension Bayesian optimisation implementation.
"""

import warnings
import traceback
import numpy as np

import sys
sys.path.append('../pybop')
from optimization import (
    _matern52,
    _compute_r2,
    GaussianProcess,
    bayesian_minimization,
)

# ─────────────────────────────────────────────────────────────────────────────
# Terminal formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS_TAG = f"{GREEN}{BOLD}  PASS{RESET}"
FAIL_TAG = f"{RED}{BOLD}  FAIL{RESET}"

_results: list[dict] = []   # Accumulated across all tests for the summary


def _section(title: str) -> None:
    width = 72
    print(f"\n{CYAN}{BOLD}{'═' * width}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'═' * width}{RESET}")


def _run_test(name: str, fn) -> bool:
    """
    Execute *fn()*.  Print a structured result block and return True on pass.
    """
    print(f"\n  {BOLD}▸ {name}{RESET}")
    try:
        info = fn()          # fn may return a dict of diagnostics or None
        print(f"{PASS_TAG}")
        if isinstance(info, dict):
            for k, v in info.items():
                print(f"      {YELLOW}{k}{RESET}: {v}")
        _results.append({"name": name, "passed": True})
        return True
    except Exception as exc:          # noqa: BLE001
        print(f"{FAIL_TAG}")
        print(f"      {RED}Error : {exc}{RESET}")
        # Print the traceback indented so it stays visually grouped
        tb_lines = traceback.format_exc().splitlines()
        for line in tb_lines:
            print(f"        {RED}{line}{RESET}")
        _results.append({"name": name, "passed": False})
        return False


def _summary() -> None:
    width  = 72
    passed = sum(r["passed"] for r in _results)
    total  = len(_results)
    failed = total - passed

    print(f"\n{CYAN}{BOLD}{'═' * width}{RESET}")
    print(f"{CYAN}{BOLD}  SUMMARY{RESET}")
    print(f"{CYAN}{BOLD}{'═' * width}{RESET}")
    for r in _results:
        tag  = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        mark = "✓" if r["passed"] else "✗"
        print(f"  {tag}  {mark}  {r['name']}")

    print(f"\n  {BOLD}Passed : {GREEN}{passed}{RESET} / {total}")
    if failed:
        print(f"  {BOLD}Failed : {RED}{failed}{RESET} / {total}")
    print(f"{CYAN}{BOLD}{'═' * width}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_gp(periodic_dims=None, periods=None,
             n_restarts=3, n_screen=12, random_state=0, **kw):
    return GaussianProcess(
        periodic_dims=periodic_dims,
        periods=periods,
        n_restarts=n_restarts,
        n_screen=n_screen,
        random_state=random_state,
        **kw,
    )


def _finite_diff_gradient(f, x, eps=1e-5):
    x    = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm  = x.copy(), x.copy()
        xp[i]  += eps
        xm[i]  -= eps
        grad[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return grad


def _assert_allclose(actual, desired, rtol=1e-8, atol=1e-10, label=""):
    """Thin wrapper that raises AssertionError with a numeric summary."""
    actual  = np.asarray(actual,  dtype=float)
    desired = np.asarray(desired, dtype=float)
    diff    = np.abs(actual - desired)
    max_diff = float(diff.max())
    max_rel  = float((diff / (np.abs(desired) + 1e-30)).max())
    if not np.allclose(actual, desired, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label}  max|Δ|={max_diff:.3e}  max|Δ|/|ref|={max_rel:.3e}  "
            f"(tol: rtol={rtol}, atol={atol})"
        )
    return {"max_abs_error": f"{max_diff:.3e}",
            "max_rel_error": f"{max_rel:.3e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Kernel periodicity
# ─────────────────────────────────────────────────────────────────────────────

def _kernel_periodicity_run(label, d, periodic_dims, periods_orig,
                            ls_scalar, var, X_std=None,
                            n_pts=20, rng_seed=42):
    rng     = np.random.RandomState(rng_seed)
    X_std   = X_std if X_std is not None else np.ones(d)
    periods_norm = np.ones(d)
    for i, k in enumerate(periodic_dims):
        periods_norm[k] = periods_orig[i] / X_std[k]
    ls_arr  = ls_scalar * np.ones(d)
    X1      = rng.randn(n_pts, d)

    max_err = 0.0
    for i, k in enumerate(periodic_dims):
        T_k   = periods_norm[k]
        X2    = X1.copy()
        X2[:, k] += T_k
        K_base    = _matern52(X1, X1, ls_arr, var, periodic_dims, periods_norm)
        K_shifted = _matern52(X1, X2, ls_arr, var, periodic_dims, periods_norm)
        err       = np.abs(np.diag(K_shifted) - np.diag(K_base)).max()
        max_err   = max(max_err, err)
        if err > 1e-8:
            raise AssertionError(
                f"[{label}] dim {k}: K(x, x+T) ≠ K(x,x)  "
                f"max diagonal error = {err:.3e}"
            )
    return {"label": label,
            "periodic_dims": periodic_dims,
            "max_diagonal_error": f"{max_err:.3e}"}


def test_kernel_periodicity_1d():
    return _kernel_periodicity_run(
        "1-D periodic", d=1,
        periodic_dims=[0], periods_orig=[2.0],
        ls_scalar=0.5, var=1.0)


def test_kernel_periodicity_2d_both():
    return _kernel_periodicity_run(
        "2-D both periodic", d=2,
        periodic_dims=[0, 1], periods_orig=[2.0, 3.0],
        ls_scalar=0.5, var=1.5)


def test_kernel_periodicity_2d_one():
    return _kernel_periodicity_run(
        "2-D dim-1 periodic", d=2,
        periodic_dims=[1], periods_orig=[2 * np.pi],
        ls_scalar=1.0, var=1.0)


def test_kernel_half_period_translation():
    """K(x, x+T/2) invariant under translation by T/2."""
    d, T = 1, 2 * np.pi
    ls   = np.array([1.0])
    pn   = np.array([T])
    rng  = np.random.RandomState(7)
    X1   = rng.randn(15, d)
    X2   = X1 + T / 2
    K_a  = _matern52(X1,          X2,          ls, 1.0, [0], pn)
    K_b  = _matern52(X1 + T / 2, X2 + T / 2, ls, 1.0, [0], pn)
    info = _assert_allclose(K_a, K_b, rtol=1e-8, atol=1e-10,
                            label="half-period translation")
    info["shift"] = "T/2"
    return info


# ─────────────────────────────────────────────────────────────────────────────
# 2. Kernel symmetry
# ─────────────────────────────────────────────────────────────────────────────

def test_kernel_symmetry_no_periodic():
    rng = np.random.RandomState(0)
    X   = rng.randn(10, 3)
    ls  = np.array([0.5, 1.0, 2.0])
    K   = _matern52(X, X, ls, 1.2)
    asym = float(np.abs(K - K.T).max())
    if asym > 1e-12:
        raise AssertionError(f"Kernel not symmetric: max|K - Kᵀ| = {asym:.3e}")
    return {"max_asymmetry": f"{asym:.3e}", "shape": str(K.shape)}


def test_kernel_symmetry_with_periodic():
    rng  = np.random.RandomState(1)
    X    = rng.randn(12, 3)
    ls   = np.array([0.5, 1.0, 2.0])
    pn   = np.array([1.0, 2 * np.pi, 1.0])
    K    = _matern52(X, X, ls, 1.0, [0, 1], pn)
    asym = float(np.abs(K - K.T).max())
    if asym > 1e-12:
        raise AssertionError(f"Kernel not symmetric: max|K - Kᵀ| = {asym:.3e}")
    return {"max_asymmetry": f"{asym:.3e}", "periodic_dims": [0, 1]}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kernel positive-definiteness
# ─────────────────────────────────────────────────────────────────────────────

def _check_pd(K, label):
    eigvals = np.linalg.eigvalsh(K)
    min_eig = float(eigvals.min())
    if min_eig < -1e-8:
        raise AssertionError(
            f"[{label}] Kernel not PD: min eigenvalue = {min_eig:.3e}"
        )
    return {"label": label,
            "min_eigenvalue": f"{min_eig:.3e}",
            "shape": str(K.shape)}


def test_pd_pure_standard():
    rng = np.random.RandomState(2)
    X   = rng.randn(20, 4)
    K   = _matern52(X, X, np.ones(4), 1.0)
    return _check_pd(K, "pure standard")


def test_pd_pure_periodic():
    rng = np.random.RandomState(3)
    X   = rng.randn(20, 2)
    pn  = np.array([2.0, np.pi])
    K   = _matern52(X, X, np.ones(2), 1.0, [0, 1], pn)
    return _check_pd(K, "pure periodic")


def test_pd_mixed():
    rng = np.random.RandomState(4)
    X   = rng.randn(15, 3)
    pn  = np.array([2.0, 1.0, 1.0])
    K   = _matern52(X, X, np.ones(3), 1.0, [1], pn)
    return _check_pd(K, "mixed (dim-1 periodic)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gradient check — log-marginal-likelihood
# ─────────────────────────────────────────────────────────────────────────────

def _lml_gradient_run(label, d, periodic_dims, periods, n=12, seed=0):
    rng = np.random.RandomState(seed)
    X   = rng.uniform(0, 2 * np.pi, (n, d))
    y   = np.sin(X[:, 0]) + 0.1 * rng.randn(n)

    gp = _make_gp(periodic_dims=periodic_dims, periods=periods,
                  deterministic=False)
    # Bootstrap normalisation without running the full hyperparameter optimiser
    gp.X_mean_ = X.mean(0);  gp.X_std_ = X.std(0) + 1e-12
    gp.y_mean_ = float(y.mean()); gp.y_std_ = float(y.std() + 1e-12)
    gp._periods_norm = np.ones(d)
    for i, k in enumerate(periodic_dims):
        gp._periods_norm[k] = periods[i] / gp.X_std_[k]
    gp._mean_dims = [k for k in range(d) if k not in periodic_dims]
    n_md = len(gp._mean_dims)
    gp._beta = np.zeros(n_md + 1)
    Xn = gp._normalise_X(X); yn = gp._normalise_y(y)
    gp._beta[0] = yn.mean()
    gp.X_train_ = Xn; gp.y_train_ = yn - gp._mean(Xn)
    gp.length_scale_ = np.ones(d)

    n_params  = d + 2
    log_p0    = np.zeros(n_params)
    _, grad_a = gp._log_marginal_likelihood(log_p0)

    def scalar_lml(lp):
        return gp._log_marginal_likelihood(lp)[0]

    grad_fd  = _finite_diff_gradient(scalar_lml, log_p0, eps=1e-4)
    max_diff = float(np.abs(grad_a - grad_fd).max())
    max_rel  = float((np.abs(grad_a - grad_fd)
                      / (np.abs(grad_fd) + 1e-30)).max())

    if not np.allclose(grad_a, grad_fd, rtol=1e-3, atol=1e-5):
        raise AssertionError(
            f"[{label}] LML gradient mismatch  "
            f"max|Δ|={max_diff:.3e}  max_rel={max_rel:.3e}\n"
            f"  analytic : {grad_a}\n"
            f"  finite-d : {grad_fd}"
        )
    return {"label": label,
            "n_params": n_params,
            "max_abs_error": f"{max_diff:.3e}",
            "max_rel_error": f"{max_rel:.3e}"}


def test_lml_grad_no_periodic():
    return _lml_gradient_run("no periodic", d=2,
                             periodic_dims=[], periods=[])

def test_lml_grad_dim0_periodic():
    return _lml_gradient_run("dim-0 periodic", d=2,
                             periodic_dims=[0], periods=[2 * np.pi])

def test_lml_grad_both_periodic():
    return _lml_gradient_run("both periodic", d=2,
                             periodic_dims=[0, 1],
                             periods=[2 * np.pi, 2 * np.pi])

def test_lml_grad_3d_mixed():
    return _lml_gradient_run("3-D mixed (dim-1 periodic)", d=3,
                             periodic_dims=[1], periods=[2 * np.pi])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gradient check — predict_with_gradients
# ─────────────────────────────────────────────────────────────────────────────

def _build_fitted_gp(d, periodic_dims, periods, n=20, seed=1):
    rng = np.random.RandomState(seed)
    X   = rng.uniform(0, 2 * np.pi, (n, d))
    y   = (np.cos(X[:, 0])
           + (np.sin(X[:, 1]) if d > 1 else 0.0)
           + 0.05 * rng.randn(n))
    gp  = _make_gp(periodic_dims=periodic_dims, periods=periods,
                   n_restarts=2, n_screen=6)
    gp.fit(X, y)
    return gp


def _check_predict_gradients(gp, x_test, eps=1e-5, rtol=1e-3, atol=1e-5):
    mu, sigma, dmu, dsigma = gp.predict_with_gradients(x_test)

    def mu_fn(x):
        return float(gp.predict(x[None, :], return_std=False)[0])

    def sigma_fn(x):
        _, s = gp.predict(x[None, :], return_std=True)
        return float(s[0])

    dmu_fd    = _finite_diff_gradient(mu_fn,    x_test, eps=eps)
    dsigma_fd = _finite_diff_gradient(sigma_fn, x_test, eps=eps)

    err_mu    = float(np.abs(dmu    - dmu_fd).max())
    err_sigma = float(np.abs(dsigma - dsigma_fd).max())

    if not np.allclose(dmu, dmu_fd, rtol=rtol, atol=atol):
        raise AssertionError(
            f"dmu/dx mismatch at x={x_test}  max|Δ|={err_mu:.3e}\n"
            f"  analytic : {dmu}\n"
            f"  finite-d : {dmu_fd}"
        )
    if not np.allclose(dsigma, dsigma_fd, rtol=rtol, atol=atol):
        raise AssertionError(
            f"dsigma/dx mismatch at x={x_test}  max|Δ|={err_sigma:.3e}\n"
            f"  analytic : {dsigma}\n"
            f"  finite-d : {dsigma_fd}"
        )
    return err_mu, err_sigma


def test_predict_grad_1d_periodic():
    gp     = _build_fitted_gp(1, [0], [2 * np.pi])
    x_test = np.array([1.23])
    e_mu, e_s = _check_predict_gradients(gp, x_test)
    return {"x_test": x_test,
            "max_err_dmu": f"{e_mu:.3e}",
            "max_err_dsigma": f"{e_s:.3e}"}


def test_predict_grad_2d_no_periodic():
    gp     = _build_fitted_gp(2, [], [])
    x_test = np.array([1.0, 2.0])
    e_mu, e_s = _check_predict_gradients(gp, x_test)
    return {"x_test": x_test.tolist(),
            "max_err_dmu": f"{e_mu:.3e}",
            "max_err_dsigma": f"{e_s:.3e}"}


def test_predict_grad_2d_one_periodic():
    gp     = _build_fitted_gp(2, [0], [2 * np.pi])
    x_test = np.array([0.5, 1.5])
    e_mu, e_s = _check_predict_gradients(gp, x_test)
    return {"x_test": x_test.tolist(),
            "max_err_dmu": f"{e_mu:.3e}",
            "max_err_dsigma": f"{e_s:.3e}"}


def test_predict_grad_2d_both_periodic():
    gp     = _build_fitted_gp(2, [0, 1], [2 * np.pi, 2 * np.pi])
    x_test = np.array([1.1, 2.2])
    e_mu, e_s = _check_predict_gradients(gp, x_test)
    return {"x_test": x_test.tolist(),
            "max_err_dmu": f"{e_mu:.3e}",
            "max_err_dsigma": f"{e_s:.3e}"}


def test_predict_grad_3d_mixed():
    gp     = _build_fitted_gp(3, [1], [2 * np.pi])
    x_test = np.array([0.3, 1.7, 2.5])
    e_mu, e_s = _check_predict_gradients(gp, x_test)
    return {"x_test": x_test.tolist(),
            "max_err_dmu": f"{e_mu:.3e}",
            "max_err_dsigma": f"{e_s:.3e}"}


def test_predict_grad_multiple_random_points():
    """Run gradient check at 8 random test points."""
    gp  = _build_fitted_gp(2, [0], [2 * np.pi])
    rng = np.random.RandomState(99)
    worst_mu = worst_s = 0.0
    for _ in range(8):
        x_test   = rng.uniform(0, 2 * np.pi, 2)
        e_mu, e_s = _check_predict_gradients(gp, x_test)
        worst_mu  = max(worst_mu, e_mu)
        worst_s   = max(worst_s,  e_s)
    return {"n_points_checked": 8,
            "worst_err_dmu": f"{worst_mu:.3e}",
            "worst_err_dsigma": f"{worst_s:.3e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Linear mean function — periodic dims excluded
# ─────────────────────────────────────────────────────────────────────────────

def test_mean_all_periodic():
    """All dims periodic → beta has shape (1,) — intercept only."""
    d   = 2
    rng = np.random.RandomState(5)
    X   = rng.uniform(0, 2 * np.pi, (40, d))
    y   = np.cos(X[:, 0]) + np.sin(X[:, 1]) + 0.05 * rng.randn(40)
    gp  = _make_gp(periodic_dims=[0, 1], periods=[2 * np.pi, 2 * np.pi])
    gp.fit(X, y)
    if gp._beta.shape != (1,):
        raise AssertionError(
            f"Expected beta shape (1,), got {gp._beta.shape}"
        )
    if gp._mean_dims != []:
        raise AssertionError(
            f"Expected mean_dims=[], got {gp._mean_dims}"
        )
    return {"beta_shape": str(gp._beta.shape),
            "beta_value (intercept)": f"{gp._beta[0]:.4f}",
            "mean_dims": gp._mean_dims}


def test_mean_mixed_periodic():
    """1 periodic + 1 standard dim → beta has shape (2,)."""
    d   = 2
    rng = np.random.RandomState(6)
    X   = rng.uniform(0, 3, (40, d))
    y   = X[:, 0] + np.cos(2 * np.pi * X[:, 1] / 2.0) + 0.05 * rng.randn(40)
    gp  = _make_gp(periodic_dims=[1], periods=[2.0])
    gp.fit(X, y)
    if gp._beta.shape != (2,):
        raise AssertionError(
            f"Expected beta shape (2,), got {gp._beta.shape}"
        )
    if gp._mean_dims != [0]:
        raise AssertionError(
            f"Expected mean_dims=[0], got {gp._mean_dims}"
        )
    return {"beta_shape": str(gp._beta.shape),
            "intercept": f"{gp._beta[0]:.4f}",
            "slope_dim0": f"{gp._beta[1]:.4f}",
            "mean_dims": gp._mean_dims}


def test_mean_no_periodic():
    """No periodic dims → beta has shape (d+1,)."""
    d   = 2
    rng = np.random.RandomState(7)
    X   = rng.uniform(0, 3, (40, d))
    y   = X[:, 0] + 0.5 * X[:, 1] + 0.05 * rng.randn(40)
    gp  = _make_gp()
    gp.fit(X, y)
    if gp._beta.shape != (d + 1,):
        raise AssertionError(
            f"Expected beta shape ({d+1},), got {gp._beta.shape}"
        )
    return {"beta_shape": str(gp._beta.shape),
            "intercept": f"{gp._beta[0]:.4f}",
            "slopes": [f"{b:.4f}" for b in gp._beta[1:]],
            "mean_dims": gp._mean_dims}


# ─────────────────────────────────────────────────────────────────────────────
# 7. GP fit + predict sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_gp_interpolation_periodic():
    """Posterior mean should recover noise-free training targets."""
    X   = np.linspace(0, 2 * np.pi, 12)[:, None]
    y   = np.cos(X[:, 0])
    gp  = _make_gp(periodic_dims=[0], periods=[2 * np.pi], deterministic=True)
    gp.fit(X, y)
    mu  = gp.predict(X)
    max_err = float(np.abs(mu - y).max())
    if max_err > 1e-3:
        raise AssertionError(
            f"GP interpolation error too large: max|μ-y| = {max_err:.4f}"
        )
    return {"max_interpolation_error": f"{max_err:.3e}",
            "n_train": len(y)}


def test_gp_interpolation_standard():
    X   = np.linspace(0, 5, 15)[:, None]
    y   = np.sin(X[:, 0])
    gp  = _make_gp(deterministic=True)
    gp.fit(X, y)
    mu  = gp.predict(X)
    max_err = float(np.abs(mu - y).max())
    if max_err > 1e-3:
        raise AssertionError(
            f"GP interpolation error too large: max|μ-y| = {max_err:.4f}"
        )
    return {"max_interpolation_error": f"{max_err:.3e}",
            "n_train": len(y)}


def test_gp_std_at_training_points():
    """Posterior std should be near zero at noise-free training locations."""
    X   = np.linspace(0, 2 * np.pi, 10)[:, None]
    y   = np.cos(X[:, 0])
    gp  = _make_gp(periodic_dims=[0], periods=[2 * np.pi], deterministic=True)
    gp.fit(X, y)
    _, sigma = gp.predict(X, return_std=True)
    max_s = float(sigma.max())
    if max_s > 1e-2:
        raise AssertionError(
            f"Posterior std too large at training points: {max_s:.4f}"
        )
    return {"max_sigma_at_train": f"{max_s:.3e}"}


def test_gp_std_larger_away_from_data():
    """Uncertainty must be larger in unexplored regions than at training pts."""
    X_tr = np.array([[0.0], [6.0]])
    y_tr = np.cos(X_tr[:, 0])
    gp   = _make_gp(periodic_dims=[0], periods=[2 * np.pi], deterministic=True)
    gp.fit(X_tr, y_tr)

    X_mid          = np.array([[np.pi]])
    _, s_mid       = gp.predict(X_mid,  return_std=True)
    _, s_train     = gp.predict(X_tr,   return_std=True)
    s_mid_val      = float(s_mid[0])
    s_train_max    = float(s_train.max())

    if s_mid_val <= s_train_max:
        raise AssertionError(
            f"Expected σ(x_mid)={s_mid_val:.4f} > σ_train_max={s_train_max:.4f}"
        )
    return {"sigma_at_midpoint": f"{s_mid_val:.4f}",
            "sigma_max_at_train": f"{s_train_max:.4f}",
            "ratio": f"{s_mid_val / (s_train_max + 1e-30):.2f}×"}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Periodic wrapping — predict at x and x + T gives same result
# ─────────────────────────────────────────────────────────────────────────────

def _periodic_wrapping_run(label, d, periodic_dims, periods, x_test, seed=0):
    rng = np.random.RandomState(seed)
    X   = rng.uniform(0, periods[0], (20, d))
    y   = np.cos(2 * np.pi * X[:, periodic_dims[0]] / periods[0])
    y  += 0.05 * rng.randn(20)
    gp  = _make_gp(periodic_dims=periodic_dims, periods=periods)
    gp.fit(X, y)

    x1      = np.array(x_test, dtype=float)
    x2      = x1.copy();  x2[periodic_dims[0]] += periods[0]
    mu1, s1 = gp.predict(x1[None, :], return_std=True)
    mu2, s2 = gp.predict(x2[None, :], return_std=True)

    err_mu = abs(float(mu1[0]) - float(mu2[0]))
    err_s  = abs(float(s1[0])  - float(s2[0]))
    if err_mu > 1e-5 or err_s > 1e-5:
        raise AssertionError(
            f"[{label}] Wrapping failed: Δμ={err_mu:.3e}, Δσ={err_s:.3e}"
        )
    return {"label": label,
            "mu_at_x": f"{float(mu1[0]):.6f}",
            "mu_at_x+T": f"{float(mu2[0]):.6f}",
            "delta_mu": f"{err_mu:.3e}",
            "delta_sigma": f"{err_s:.3e}"}


def test_wrapping_1d():
    return _periodic_wrapping_run(
        "1-D", d=1, periodic_dims=[0], periods=[2 * np.pi],
        x_test=[1.2])

def test_wrapping_2d_dim1():
    return _periodic_wrapping_run(
        "2-D dim-1 periodic", d=2, periodic_dims=[1], periods=[2 * np.pi],
        x_test=[0.5, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# 9. BO run — 1-D periodic objective
# ─────────────────────────────────────────────────────────────────────────────

def test_bo_1d_periodic():
    """
    Minimise f(x) = cos(x) on [0, 4π].  True minimum = -1 at x = π, 3π.
    Expect y_best ≤ -0.95 within 30 calls.
    """
    def f(x):
        return float(np.cos(x[0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bayesian_minimization(
            f,
            bounds=[(0.0, 4 * np.pi)],
            n_initial=8,
            n_calls=30,
            periodic_dims=[0],
            periods=[2 * np.pi],
            random_state=42,
            verbose=False,
            n_samples=5_000,
            n_polish=10,
            gp_kwargs={"n_restarts": 3, "n_screen": 10},
        )

    y_best = result["y_best"]
    x_best = result["x_best"]
    if y_best > -0.95:
        raise AssertionError(
            f"BO failed to find cos minimum: y_best = {y_best:.4f} > -0.95"
        )
    return {"y_best": f"{y_best:.6f}",
            "x_best": f"{x_best[0]:.4f}",
            "true_minimum": "-1.0",
            "n_calls_used": len(result["y"])}


# ─────────────────────────────────────────────────────────────────────────────
# 10. BO run — 2-D mixed (1 periodic + 1 standard)
# ─────────────────────────────────────────────────────────────────────────────

def test_bo_mixed_2d_finds_minimum():
    """
    Minimise f(x,t) = (x-2)^2 + cos(t), periodic dim t with T=2π.
    True minimum: x*=2, t*=π (or 3π), f*=-1.
    Expect y_best ≤ -0.90 within 40 calls.
    """
    def f(v):
        x, t = v
        return (x - 2.0) ** 2 + np.cos(t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bayesian_minimization(
            f,
            bounds=[(0.0, 4.0), (0.0, 4 * np.pi)],
            n_initial=8,
            n_calls=40,
            periodic_dims=[1],
            periods=[2 * np.pi],
            random_state=7,
            verbose=False,
            n_samples=10_000,
            n_polish=10,
            gp_kwargs={"n_restarts": 3, "n_screen": 10},
        )

    y_best = result["y_best"]
    x_opt, t_opt = result["x_best"]
    if y_best > -0.90:
        raise AssertionError(
            f"BO did not find the mixed minimum: y_best={y_best:.4f} > -0.90"
        )
    return {"y_best": f"{y_best:.6f}",
            "x_opt": f"{x_opt:.4f}  (true: 2.000)",
            "t_opt": f"{t_opt:.4f}  (true: π=3.1416 or 3π=9.4248)",
            "true_minimum": "-1.0"}


def test_bo_mixed_2d_x_near_optimum():
    """
    Same objective; check that x_best is numerically close to (2, π mod 2π).
    Uses more calls for a tighter result.
    """
    def f(v):
        x, t = v
        return (x - 2.0) ** 2 + np.cos(t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bayesian_minimization(
            f,
            bounds=[(0.0, 4.0), (0.0, 4 * np.pi)],
            n_initial=8,
            n_calls=50,
            periodic_dims=[1],
            periods=[2 * np.pi],
            random_state=13,
            verbose=False,
            n_samples=10_000,
            n_polish=10,
            gp_kwargs={"n_restarts": 3, "n_screen": 10},
        )

    x_opt, t_opt = result["x_best"]
    err_x  = abs(x_opt - 2.0)
    t_mod  = t_opt % (2 * np.pi)
    err_t  = abs(t_mod - np.pi)

    if err_x > 0.2:
        raise AssertionError(
            f"x not near 2.0: x_opt={x_opt:.4f}, err={err_x:.4f}"
        )
    if err_t > 0.3:
        raise AssertionError(
            f"t not near π (mod 2π): t_mod={t_mod:.4f}, err={err_t:.4f}"
        )
    return {"x_opt": f"{x_opt:.4f}  (err={err_x:.4f})",
            "t_opt": f"{t_opt:.4f}  (t mod 2π = {t_mod:.4f}, err={err_t:.4f})",
            "y_best": f"{result['y_best']:.6f}"}


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Section 1 ─────────────────────────────────────────────────────────
    _section("1 · Kernel Periodicity  —  K(x, x+T) = K(x, x)")
    _run_test("1-D periodic kernel",               test_kernel_periodicity_1d)
    _run_test("2-D both dims periodic",            test_kernel_periodicity_2d_both)
    _run_test("2-D dim-1 periodic only",           test_kernel_periodicity_2d_one)
    _run_test("Half-period translation invariance", test_kernel_half_period_translation)

    # ── Section 2 ─────────────────────────────────────────────────────────
    _section("2 · Kernel Symmetry  —  K(X,X) = K(X,X)ᵀ")
    _run_test("Symmetry, no periodic dims",         test_kernel_symmetry_no_periodic)
    _run_test("Symmetry, with periodic dims",       test_kernel_symmetry_with_periodic)

    # ── Section 3 ─────────────────────────────────────────────────────────
    _section("3 · Kernel Positive-Definiteness")
    _run_test("PD — pure standard dims",            test_pd_pure_standard)
    _run_test("PD — pure periodic dims",            test_pd_pure_periodic)
    _run_test("PD — mixed dims",                    test_pd_mixed)

    # ── Section 4 ─────────────────────────────────────────────────────────
    _section("4 · Log-Marginal-Likelihood Gradient  (analytic vs finite-diff)")
    _run_test("LML grad — no periodic dims",        test_lml_grad_no_periodic)
    _run_test("LML grad — dim-0 periodic",          test_lml_grad_dim0_periodic)
    _run_test("LML grad — both dims periodic",      test_lml_grad_both_periodic)
    _run_test("LML grad — 3-D, dim-1 periodic",     test_lml_grad_3d_mixed)

    # ── Section 5 ─────────────────────────────────────────────────────────
    _section("5 · predict_with_gradients  (analytic vs finite-diff)")
    _run_test("Predict grad — 1-D periodic",        test_predict_grad_1d_periodic)
    _run_test("Predict grad — 2-D no periodic",     test_predict_grad_2d_no_periodic)
    _run_test("Predict grad — 2-D dim-0 periodic",  test_predict_grad_2d_one_periodic)
    _run_test("Predict grad — 2-D both periodic",   test_predict_grad_2d_both_periodic)
    _run_test("Predict grad — 3-D mixed",           test_predict_grad_3d_mixed)
    _run_test("Predict grad — 8 random points",     test_predict_grad_multiple_random_points)

    # ── Section 6 ─────────────────────────────────────────────────────────
    _section("6 · Linear Mean Function  —  Periodic Dims Excluded")
    _run_test("All dims periodic  → beta shape (1,)",  test_mean_all_periodic)
    _run_test("Mixed dims         → beta shape (2,)",  test_mean_mixed_periodic)
    _run_test("No periodic dims   → beta shape (d+1,)", test_mean_no_periodic)

    # ── Section 7 ─────────────────────────────────────────────────────────
    _section("7 · GP Fit & Predict Sanity")
    _run_test("Interpolation — periodic kernel",    test_gp_interpolation_periodic)
    _run_test("Interpolation — standard kernel",    test_gp_interpolation_standard)
    _run_test("Posterior std ≈ 0 at training pts",  test_gp_std_at_training_points)
    _run_test("Posterior std larger away from data", test_gp_std_larger_away_from_data)

    # ── Section 8 ─────────────────────────────────────────────────────────
    _section("8 · Periodic Wrapping  —  predict(x) = predict(x + T)")
    _run_test("Wrapping — 1-D",                     test_wrapping_1d)
    _run_test("Wrapping — 2-D, dim-1 periodic",     test_wrapping_2d_dim1)

    # ── Section 9 ─────────────────────────────────────────────────────────
    _section("9 · End-to-End BO  —  1-D Periodic Objective  f(x)=cos(x)")
    _run_test("BO finds cos minimum (y ≤ -0.95)",   test_bo_1d_periodic)

    # ── Section 10 ────────────────────────────────────────────────────────
    _section("10 · End-to-End BO  —  2-D Mixed Objective  f(x,t)=(x-2)²+cos(t)")
    _run_test("BO finds minimum (y ≤ -0.90)",        test_bo_mixed_2d_finds_minimum)
    _run_test("x_best near true optimum (2, π)",    test_bo_mixed_2d_x_near_optimum)

    # ── Summary ───────────────────────────────────────────────────────────
    _summary()


if __name__ == "__main__":
    main()
