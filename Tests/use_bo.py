import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
from pybop import *
import skopt
import time
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def sphere(d, seed=None):
    """
    Sphere: f(x) = sum x_i^2.
    Smooth, convex, isotropic — the simplest possible benchmark.
    Both GP and skopt should solve this easily; mainly checks basic
    convergence and that the ARD length scales are tuned correctly.
    Global minimum: x* = 0,  f* = 0.
    """
    bounds     = [(-5.0, 5.0)] * d
    x_true_min = np.zeros(d)
    y_true_min = 0.0

    def f(x):
        x = np.asarray(x)
        return float(np.sum(x * x))

    return f, bounds, x_true_min, y_true_min


def rotated_ellipsoid(d, condition=1e3, seed=0):
    """
    Rotated ellipsoid: a quadratic bowl with random orientation and a
    large condition number.  ARD length scales must adapt to the principal
    axes; isotropic kernels struggle here.
    Global minimum: x* = random centre,  f* = 0.
    """
    rng        = np.random.RandomState(seed)
    bounds     = [(-5.0, 5.0)] * d
    x_true_min = rng.uniform(-3.0, 3.0, size=d)
    y_true_min = 0.0
    Q, _       = np.linalg.qr(rng.randn(d, d))
    scales     = np.geomspace(1, condition, d)

    def f(x):
        x = np.asarray(x)
        y = Q @ (x - x_true_min)
        return float(np.sum(scales * y ** 2))

    return f, bounds, x_true_min, y_true_min


def ackley(d, a=20.0, b=0.2, c=2 * np.pi, seed=None):
    """
    Ackley: many shallow local minima surrounding a deep global minimum.
    Classic test of global search capability.
    Global minimum: x* = 0,  f* = 0.
    """
    bounds     = [(-5.0, 5.0)] * d
    x_true_min = np.zeros(d)
    y_true_min = 0.0

    def f(x):
        x = np.asarray(x)
        return float(
            a * (1 - np.exp(-b * np.sqrt(np.sum(x * x) / d)))
            + np.e
            - np.exp(np.sum(np.cos(c * x)) / d)
        )

    return f, bounds, x_true_min, y_true_min


def rosenbrock(d, seed=None):
    """
    Rosenbrock (banana): narrow curved valley; gradient points mostly
    perpendicular to the valley floor.  Tests exploitation along a
    poorly-conditioned manifold.
    Global minimum: x* = 1,  f* = 0.
    """
    bounds     = [(-2.0, 2.0)] * d
    x_true_min = np.ones(d)
    y_true_min = 0.0

    def f(x):
        x = np.asarray(x)
        return float(np.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2
            + (1 - x[:-1]) ** 2
        ))

    return f, bounds, x_true_min, y_true_min


def schwefel(d, seed=None):
    """
    Schwefel: highly multi-modal; the global minimum is far from the next
    best local minima, and the function is deceptive (local gradients point
    away from the global minimum).  Hard for any surrogate method.
    Global minimum: x* ≈ 420.97,  f* = 0.
    """
    bounds     = [(-500.0, 500.0)] * d
    x_true_min = np.full(d, 420.968746)
    y_true_min = 0.0

    def f(x):
        x = np.asarray(x)
        return float(418.9828872724337 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    return f, bounds, x_true_min, y_true_min


def rastrigin(d, seed=None):
    """
    Rastrigin: regular grid of local minima superimposed on a parabola.
    The cosine term makes the landscape look globally quadratic but locally
    highly structured.  Tests ability to avoid the many local minima.
    Global minimum: x* = 0,  f* = 0.
    """
    bounds     = [(-5.12, 5.12)] * d
    x_true_min = np.zeros(d)
    y_true_min = 0.0

    def f(x):
        x = np.asarray(x)
        return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    return f, bounds, x_true_min, y_true_min


def levy(d, seed=None):
    """
    Levy: another multi-modal function with a single global minimum.
    Constructed from sin and cos terms at varying frequencies; harder than
    Rastrigin because the local minima are not on a regular grid.
    Global minimum: x* = 1,  f* = 0.
    """
    bounds     = [(-5.0, 5.0)] * d
    x_true_min = np.ones(d)
    y_true_min = 0.0

    def f(x):
        x  = np.asarray(x, dtype=float)
        w  = 1.0 + (x - 1.0) / 4.0
        t1 = np.sin(np.pi * w[0]) ** 2
        td = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        ti = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        return float(t1 + ti + td)

    return f, bounds, x_true_min, y_true_min


def griewank(d, seed=None):
    """
    Griewank: product of cosines modulates a parabolic bowl.  Similar to
    Rastrigin but the local-minima structure depends on ALL dimensions
    simultaneously (via the product), so the function is non-separable.
    Tests whether the GP captures cross-dimensional correlations.
    Global minimum: x* = 0,  f* = 0.
    """
    bounds     = [(-10.0, 10.0)] * d
    x_true_min = np.zeros(d)
    y_true_min = 0.0

    def f(x):
        x   = np.asarray(x, dtype=float)
        idx = np.arange(1, d + 1, dtype=float)
        return float(
            1.0
            + np.sum(x ** 2) / 4000.0
            - np.prod(np.cos(x / np.sqrt(idx)))
        )

    return f, bounds, x_true_min, y_true_min


def dixon_price(d, seed=None):
    """
    Dixon-Price: a single narrow curved valley similar to Rosenbrock but
    with a different algebraic structure.  The global minimum lies on a
    1-D manifold inside the d-dimensional box, requiring precise exploitation.
    Global minimum: x_i* = 2^(-(2^i-2)/2^i),  f* = 0.
    """
    bounds     = [(-10.0, 10.0)] * d
    i          = np.arange(2, d + 1, dtype=float)
    x_true_min = np.concatenate([[1.0],
                                  2.0 ** (-(2.0 ** i - 2.0) / 2.0 ** i)])
    y_true_min = 0.0

    def f(x):
        x = np.asarray(x, dtype=float)
        t1 = (x[0] - 1.0) ** 2
        ti = np.sum(i * (2.0 * x[1:] ** 2 - x[:-1]) ** 2)
        return float(t1 + ti)

    return f, bounds, x_true_min, y_true_min


def styblinski_tang(d, seed=None):
    """
    Styblinski-Tang: separable but asymmetric multi-modal function.
    Each dimension has two unequal local minima — the global minimum is
    not at a 'nice' coordinate, which prevents trivial lucky guesses.
    Global minimum: x* ≈ -2.904 in every dimension,  f* ≈ -39.166 * d.
    """
    bounds     = [(-5.0, 5.0)] * d
    x_true_min = np.full(d, -2.903534)
    y_true_min = -39.16617 * d

    def f(x):
        x = np.asarray(x, dtype=float)
        return float(0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x))

    return f, bounds, x_true_min, y_true_min


def shifted_rastrigin(d, seed=0):
    """
    Rastrigin with a random shift so the global minimum is not at the
    origin.  Prevents any algorithm from using the trivial prior that
    x*=0, making it a fairer test of genuine search capability.
    Global minimum: x* = random shift,  f* = 0.
    """
    rng        = np.random.RandomState(seed)
    shift      = rng.uniform(-3.0, 3.0, size=d)
    bounds     = [(-5.12, 5.12)] * d
    x_true_min = shift
    y_true_min = 0.0

    def f(x):
        z = np.asarray(x, dtype=float) - shift
        return float(10 * d + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z)))

    return f, bounds, x_true_min, y_true_min


def hartmann6(d=6, seed=None):
    """
    Hartmann-6: a classical 6-dimensional benchmark with 6 local minima.
    Only defined for d=6; the 'd' argument is accepted but ignored so that
    the function fits the common (func, d) interface.
    Global minimum: f* ≈ -3.32237.
    """
    assert d == 6, "Hartmann-6 is only defined for d=6."
    bounds     = [(0.0, 1.0)] * 6
    x_true_min = np.array([0.20169, 0.150011, 0.476874,
                            0.275332, 0.311652, 0.6573])
    y_true_min = -3.32237

    A = np.array([
        [10,  3,  17, 3.5, 1.7,  8],
        [0.05,10, 17, 0.1,  8,  14],
        [ 3,  3.5, 1.7,10, 17,   8],
        [17,  8,  0.05,10,  0.1, 14],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569,  124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091,  381],
    ])
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    def f(x):
        x = np.asarray(x, dtype=float)
        return float(-np.sum(alpha * np.exp(
            -np.sum(A * (x[None, :] - P) ** 2, axis=1)
        )))

    return f, bounds, x_true_min, y_true_min


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

D         = 3
N_INITIAL = 32
N_CALLS   = 100
VERBOSE   = False
PLOT_DIR  = './Plots/use_bo'

TEST_FUNCTIONS = [
    # (function,           label,                         d)
    (sphere,               "Sphere",                      D),
    (rotated_ellipsoid,    "Rotated ellipsoid",           D),
    (ackley,               "Ackley",                      D),
    (rosenbrock,           "Rosenbrock",                  D),
    (schwefel,             "Schwefel",                    D),
    (rastrigin,            "Rastrigin",                   D),
    (levy,                 "Levy",                        D),
    (griewank,             "Griewank",                    D),
    (dixon_price,          "Dixon-Price",                 D),
    (styblinski_tang,      "Styblinski-Tang",             D),
    (shifted_rastrigin,    "Shifted Rastrigin",           D),
    (hartmann6,            "Hartmann-6",                  6),
]

# ---------------------------------------------------------------------------
# Run all test functions
# ---------------------------------------------------------------------------

all_results = []
# (label, f, bounds, x_true_min, y_true_min, result_ours, res_skopt, rand_vals)

for func, label, d in TEST_FUNCTIONS:

    print(f"\n{'='*72}")
    print(f"  {label}   (d={d})")
    print(f"{'='*72}")

    f, bounds, x_true_min, y_true_min = func(d)
    bounds_arr = np.asarray(bounds)

    # ---- Our Bayesian minimisation ----
    t0 = time.time()
    result = bayesian_minimization(
        objective  = f,
        bounds     = bounds,
        n_initial  = N_INITIAL,
        n_calls    = N_CALLS,
        verbose    = VERBOSE,
    )
    t_ours = time.time() - t0
    gap_ours = result['y_best'] - y_true_min
    print(f"  Our BO:   y_best - y* = {gap_ours:.4e}   ({t_ours:.1f}s)")

    # ---- skopt ----
    t0 = time.time()
    res_skopt = skopt.gp_minimize(
        f,
        bounds,
        n_initial_points = N_INITIAL,
        n_calls          = N_CALLS,
    )
    t_skopt = time.time() - t0
    gap_skopt = res_skopt.fun - y_true_min
    print(f"  skopt:    y_best - y* = {gap_skopt:.4e}   ({t_skopt:.1f}s)")

    # ---- Random search ----
    rng_rand   = np.random.RandomState(42)
    rand_X     = rng_rand.uniform(bounds_arr[:, 0], bounds_arr[:, 1],
                                  size=(N_CALLS, d))
    rand_vals  = np.array([f(x) for x in rand_X])
    gap_rand   = rand_vals.min() - y_true_min
    best_rand  = rand_X[rand_vals.argmin()]
    print(f"  Random:   y_best - y* = {gap_rand:.4e}")

    # ---- Local polish from each best found ----
    print()
    candidates = {
        'Our BO':  result['x_best'],
        'skopt':   np.asarray(res_skopt.x),
        'Random':  best_rand,
    }
    for cname, x0 in candidates.items():
        lres = minimize(f, x0, method='L-BFGS-B', bounds=bounds)
        print(f"  Local polish from {cname:8s}:  "
              f"success={lres.success},  "
              f"y-y*={lres.fun - y_true_min:.3e},  "
              f"nfev={lres.nfev}")

    all_results.append((
        label, f, bounds, x_true_min, y_true_min,
        result, res_skopt, rand_vals,
    ))

# ---------------------------------------------------------------------------
# Convergence plot — one panel per function
# ---------------------------------------------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

n_funcs = len(all_results)
n_cols  = 3
n_rows  = int(np.ceil(n_funcs / n_cols))
iter_plot = 1 + np.arange(N_CALLS)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(6 * n_cols, 4 * n_rows),
                         sharex=True)
axes = axes.flatten()

for ax, (label, f, bounds, x_true_min, y_true_min,
         result, res_skopt, rand_vals) in zip(axes, all_results):

    y_ours  = np.minimum.accumulate(result['y'])      - y_true_min
    y_skopt = np.minimum.accumulate(res_skopt.func_vals) - y_true_min
    y_rand  = np.minimum.accumulate(rand_vals)        - y_true_min

    ax.plot(iter_plot, y_ours,  label='Our BO',  color='C0')
    ax.plot(iter_plot, y_skopt, label='skopt',   color='C1')
    ax.plot(iter_plot, y_rand,  label='Random',  color='C2', ls='--')

    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Function evaluation')
    ax.set_ylabel(r'$y_\mathrm{min} - y^*$')
    ax.set_yscale('log')
    ax.set_xlim(iter_plot[0], iter_plot[-1])
    ax.legend(fontsize=8)

for ax in axes[n_funcs:]:
    ax.set_visible(False)

fig.suptitle('Our BO vs skopt vs Random — convergence comparison', fontsize=13)
fig.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"OurBO_vs_skopt_vs_Random_Convergence_Comparison.png"), dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# Slice and corner plots for our BO result on each function
# ---------------------------------------------------------------------------

for label, f, bounds, x_true_min, y_true_min, result, _, _ in all_results:

    safe_label = label.replace(" ", "_").replace("-", "_").lower()

    plot_slices(result, bounds, objective=f, x_true_min=x_true_min)
    plt.suptitle(f"{label} — Our BO", fontsize=10)
    plt.savefig(os.path.join(PLOT_DIR, f"{safe_label}_slices.png"), dpi=150, bbox_inches="tight")

    plot_corner(result, bounds, x_true_min=x_true_min)
    plt.suptitle(f"{label} — Our BO", fontsize=10)
    plt.savefig(os.path.join(PLOT_DIR, f"{safe_label}_corner.png"), dpi=150, bbox_inches="tight")
    plt.close("all")

print(f"\nAll plots saved to:  {os.path.abspath(PLOT_DIR)}")
