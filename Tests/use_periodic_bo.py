import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
from pybop import *
import time

# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def sum_cosine(d, seed=None):
    """
    Purely periodic. Sum of cosines shifted by a random phase.
    The periodic GP should have a large advantage here since the
    function is exactly periodic and separable.

        f(x) = sum_i [ 1 - cos(x_i - phi_i) ]

    Global minimum: x* = phi,  f* = 0.
    Period: 2*pi in every dimension.
    """
    rng = np.random.RandomState(seed)
    bounds       = [(-np.pi, np.pi)] * d
    x_true_min   = rng.uniform(-np.pi, np.pi, size=d)
    y_true_min   = 0.0
    periodic_dims = np.arange(d)
    periods       = np.full(d, 2 * np.pi)

    def f(x):
        x = np.asarray(x)
        return float(np.sum(1 - np.cos(x - x_true_min)))

    return f, bounds, x_true_min, y_true_min, periodic_dims, periods


def cosine_mixture(d, seed=None):
    """
    Periodic with multiple local minima per period, one of which is global.
    Each dimension is a mixture of two cosines at different frequencies,
    creating several local minima within the period. The periodic GP should
    still do well since it knows the period of the envelope.

        f(x) = sum_i [ 1 - cos(x_i - phi_i)
                         + 0.3*(1 - cos(2*(x_i - phi_i))) ]

    Global minimum: x* = phi,  f* = 0.
    Period: 2*pi in every dimension.
    """
    rng = np.random.RandomState(seed)
    bounds       = [(-np.pi, np.pi)] * d
    x_true_min   = rng.uniform(-np.pi, np.pi, size=d)
    y_true_min   = 0.0
    periodic_dims = np.arange(d)
    periods       = np.full(d, 2 * np.pi)

    def f(x):
        x   = np.asarray(x)
        dx  = x - x_true_min
        return float(np.sum(
            (1 - np.cos(dx))
            + 0.3 * (1 - np.cos(2 * dx))
        ))

    return f, bounds, x_true_min, y_true_min, periodic_dims, periods


def mixed_periodic_quadratic(d, seed=None):
    """
    Mixed: half the dimensions are periodic (cosine), the other half are
    quadratic (parabola). The periodic GP should model the two regimes
    correctly, while a standard GP has to fit both with a single kernel.
    Requires d >= 2; if d is odd the last dim is quadratic.

        Periodic dims  (i < d//2):  1 - cos(x_i - phi_i)
        Quadratic dims (i >= d//2): (x_i - c_i)^2 / pi^2

    Both contributions are normalised to have the same dynamic range (~1).
    Global minimum: periodic dims at phi, quadratic dims at c.  f* = 0.
    """
    rng = np.random.RandomState(seed)
    n_periodic   = d // 2
    n_quadratic  = d - n_periodic

    bounds       = [(-np.pi, np.pi)] * d
    x_true_min   = np.zeros(d)
    x_true_min[:n_periodic]  = rng.uniform(-np.pi, np.pi, size=n_periodic)
    x_true_min[n_periodic:]  = rng.uniform(-np.pi, np.pi, size=n_quadratic)

    y_true_min   = 0.0
    periodic_dims = np.arange(n_periodic)
    periods       = np.full(n_periodic, 2 * np.pi)

    def f(x):
        x  = np.asarray(x)
        dp = x[:n_periodic]  - x_true_min[:n_periodic]
        dq = x[n_periodic:]  - x_true_min[n_periodic:]
        return float(
            np.sum(1 - np.cos(dp))
            + np.sum(dq ** 2 / np.pi ** 2)
        )

    return f, bounds, x_true_min, y_true_min, periodic_dims, periods


def rosenbrock_periodic(d, seed=None):
    """
    Rosenbrock function (a classic hard optimisation benchmark) wrapped
    through a cosine so that it becomes periodic. This tests whether the
    periodic GP can handle a non-trivial landscape that was made periodic
    artificially rather than being naturally periodic.

        g_i(x) = 1 - cos(x_i - phi_i)           (maps x -> [0,2] per dim)
        f(x)   = sum_{i<d-1} [ 100*(g_{i+1} - g_i^2)^2 + (1 - g_i)^2 ]

    Global minimum: every g_i = 1, i.e. x_i = phi_i.  f* = 0.
    Period: 2*pi in every dimension.
    """
    rng = np.random.RandomState(seed)
    bounds       = [(-np.pi, np.pi)] * d
    x_true_min   = rng.uniform(-np.pi, np.pi, size=d)
    y_true_min   = 0.0
    periodic_dims = np.arange(d)
    periods       = np.full(d, 2 * np.pi)

    def f(x):
        x  = np.asarray(x)
        g  = 1 - np.cos(x - x_true_min)          # g_i in [0, 2]
        return float(np.sum(
            100.0 * (g[1:] - g[:-1] ** 2) ** 2
            + (1.0 - g[:-1]) ** 2
        ))

    return f, bounds, x_true_min, y_true_min, periodic_dims, periods

def grid_cosine(d, seed=None):
    """
    A product (rather than sum) of cosines, creating a multiplicative
    coupling between dimensions. The global minimum is still at x* = phi
    but the landscape has a much richer structure than the separable sum:
    all saddle-point combinations of local minima in each dimension also
    become local minima of the product.

        f(x) = 1 - prod_i cos(x_i - phi_i)

    Global minimum: x* = phi,  f* = 0.
    All dimensions are periodic with period 2*pi.
    """
    rng = np.random.RandomState(seed)
    bounds       = [(-np.pi, np.pi)] * d
    x_true_min   = rng.uniform(-np.pi, np.pi, size=d)
    y_true_min   = 0.0
    periodic_dims = np.arange(d)
    periods       = np.full(d, 2 * np.pi)

    def f(x):
        x = np.asarray(x)
        return float(1.0 - np.prod(np.cos(x - x_true_min)))

    return f, bounds, x_true_min, y_true_min, periodic_dims, periods


def short_period_cosine(d, seed=None):
    """
    Purely periodic but with a shorter period (pi instead of 2*pi), so
    there are *two* full periods inside the search bounds [-pi, pi].
    This stresses the length-scale optimisation: a non-periodic GP needs a
    short length scale to track the oscillations, while the periodic GP
    encodes the period exactly.

        f(x) = sum_i [ 1 - cos(2*(x_i - phi_i)) ]

    Global minimum: x* = phi (and phi +/- pi/2, etc. — many equivalent minima).
    f* = 0.  Period: pi in every dimension.
    """
    rng = np.random.RandomState(seed)
    bounds       = [(-np.pi, np.pi)] * d
    x_true_min   = rng.uniform(-np.pi, np.pi, size=d)
    y_true_min   = 0.0
    periodic_dims = np.arange(d)
    periods       = np.full(d, np.pi)         # shorter period

    def f(x):
        x = np.asarray(x)
        return float(np.sum(1 - np.cos(2 * (x - x_true_min))))

    return f, bounds, x_true_min, y_true_min, periodic_dims, periods


# ---------------------------------------------------------------------------
# Which functions to run, and with what settings
# ---------------------------------------------------------------------------

D         = 6
N_INITIAL = 32
N_CALLS   = 100
VERBOSE   = False    # set True to see per-iteration output
SEED      = 0
PLOT_DIR  = './Plots/use_periodic_bo'

TEST_FUNCTIONS = [
    # (function,                  label,                          d)
    (sum_cosine,                 "Sum-cosine (separable)",        D),
    (cosine_mixture,             "Cosine mixture (multi-modal)",  D),
    (mixed_periodic_quadratic,   "Mixed periodic+quadratic",      D),
    (rosenbrock_periodic,        "Rosenbrock-via-cosine",         D),
    (grid_cosine,                "Product cosine (coupled)",      D),
    (short_period_cosine,        "Short-period cosine (T=pi)",    D),
]

# ---------------------------------------------------------------------------
# Run all test functions
# ---------------------------------------------------------------------------

from matplotlib import pyplot as plt

all_results = []   # (label, f, bounds, x_true_min, y_true_min, result_np, result_p)

for func, label, d in TEST_FUNCTIONS:

    print(f"\n{'='*72}")
    print(f"  {label}   (d={d})")
    print(f"{'='*72}")

    f, bounds, x_true_min, y_true_min, periodic_dims, periods = func(d, seed=SEED)

    # ---- Non-periodic ----
    t0 = time.time()
    result_np = bayesian_minimization(
        objective     = f,
        bounds        = bounds,
        n_initial     = N_INITIAL,
        n_calls       = N_CALLS,
        verbose       = VERBOSE,
    )
    t_np = time.time() - t0
    gap_np = result_np['y_best'] - y_true_min
    print(f"  Non-periodic:  y_best - y* = {gap_np:.4e}   ({t_np:.1f}s)")

    # ---- Periodic ----
    t0 = time.time()
    result_p = bayesian_minimization(
        objective     = f,
        bounds        = bounds,
        n_initial     = N_INITIAL,
        n_calls       = N_CALLS,
        periodic_dims = periodic_dims,
        periods       = periods,
        verbose       = VERBOSE,
    )
    t_p = time.time() - t0
    gap_p = result_p['y_best'] - y_true_min
    print(f"  Periodic:      y_best - y* = {gap_p:.4e}   ({t_p:.1f}s)")

    speedup = gap_np / gap_p if gap_p > 1e-15 else float('inf')
    print(f"  Gap ratio (non-periodic / periodic): {speedup:.2f}x")

    all_results.append((label, f, bounds, x_true_min, y_true_min, result_np, result_p))


# ---------------------------------------------------------------------------
# Convergence plots — one panel per test function
# ---------------------------------------------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

n_funcs   = len(all_results)
n_cols    = 2
n_rows    = int(np.ceil(n_funcs / n_cols))
iter_plot = 1 + np.arange(N_CALLS)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(6 * n_cols, 4 * n_rows),
                         sharex=True)
axes = axes.flatten()

for ax, (label, f, bounds, x_true_min, y_true_min,
         result_np, result_p) in zip(axes, all_results):

    y_np = np.minimum.accumulate(result_np['y']) - y_true_min
    y_p  = np.minimum.accumulate(result_p ['y']) - y_true_min

    ax.plot(iter_plot, y_np, label='Non-periodic GP', color='C0')
    ax.plot(iter_plot, y_p,  label='Periodic GP',     color='C1')

    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Function evaluation number')
    ax.set_ylabel(r'$y_\mathrm{min}^\mathrm{found} - y^*$')
    ax.set_yscale('log')
    ax.set_xlim(iter_plot[0], iter_plot[-1])
    ax.legend(fontsize=8)

# Hide any unused panels
for ax in axes[n_funcs:]:
    ax.set_visible(False)

fig.suptitle('Periodic vs Non-periodic GP — convergence comparison', fontsize=12)
fig.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"Periodic_vs_NonPeriodic_GP_Convergence_Comparison.png"), dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# Slice and corner plots for each function
# ---------------------------------------------------------------------------

for label, f, bounds, x_true_min, y_true_min, result_np, result_p in all_results:

    results_dicts = {
        'Non-periodic GP': result_np,
        'Periodic GP':     result_p,
    }

    for gp_label, result in results_dicts.items():

        safe_label = f"{label} {gp_label}".replace(" ", "_").replace("-", "_").lower()

        plot_slices(result, bounds, objective=f, x_true_min=x_true_min)
        plt.suptitle(f"{label}  —  {gp_label}", fontsize=10)
        plt.savefig(os.path.join(PLOT_DIR, f"{safe_label}_slices.png"), dpi=150, bbox_inches="tight")

        plot_corner(result, bounds, x_true_min=x_true_min)
        plt.suptitle(f"{label}  —  {gp_label}", fontsize=10)
        plt.savefig(os.path.join(PLOT_DIR, f"{safe_label}_corner.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

print(f"\nAll plots saved to:  {os.path.abspath(PLOT_DIR)}")
