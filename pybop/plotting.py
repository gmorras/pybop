"""
Code for doing diagnostic plots of bayesian_minimization

Plots:
    1. Plot slices of the GP process around the found minimum
    2. Plot the points where BO has evaluated the objective function

Copyright (c) 2026 Gonzalo Morras (gonzalo.morras@aei.mpg.de)

This file is part of pybop.
Licensed under the Apache License. See the LICENSE file in the project root for details.
"""

try:
    import matplotlib.pyplot as plt
except:
    pass

import numpy as np

def plot_slices(
    result: dict,
    bounds: list[tuple[float, float]],
    objective=None,
    args:           tuple = (),
    kwargs:         dict | None = None,
    n_grid:         int = 300,
    dim_labels:     list[str] | None = None,
    figsize:        tuple | None = None,
    x_true_min:     np.ndarray | None = None,
) -> plt.Figure:
    kwargs = kwargs or {}

    def call_objective(x):
        return objective(x, *args, **kwargs)

    gp     = result["gp"]
    x_best = result["x_best"]
    y_best = result["y_best"]

    bounds_arr = np.asarray(bounds, dtype=float)
    d          = bounds_arr.shape[0]
    x_true_min = np.asarray(x_true_min, dtype=float) if x_true_min is not None else None

    if dim_labels is None:
        dim_labels = [f"x{k}" for k in range(d)]

    # One extra panel when x_true_min is given
    n_panels = d + (1 if x_true_min is not None else 0)
    ncols    = min(n_panels, 4)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize or (5.0 * ncols, 4.0 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    # --- per-dimension slice panels ---
    for k in range(d):
        ax   = axes[k // ncols][k % ncols]
        xs_k = np.linspace(bounds_arr[k, 0], bounds_arr[k, 1], n_grid)

        X_slice = np.tile(x_best, (n_grid, 1))
        X_slice[:, k] = xs_k

        mu, sigma = gp.predict(X_slice, return_std=True)

        ax.fill_between(xs_k, mu - 2 * sigma, mu + 2 * sigma,
                        alpha=0.15, color="steelblue", label=r"GP $\mu \pm 2 \sigma$")
        ax.fill_between(xs_k, mu - sigma, mu + sigma,
                        alpha=0.35, color="steelblue", label="GP $\mu \pm 1 \sigma$")
        ax.plot(xs_k, mu, color="steelblue", label="GP mean")

        if objective is not None:
            y_true = np.array([call_objective(row) for row in X_slice])
            ax.plot(xs_k, y_true, color="tomato", ls="--", label="true f")

        ax.axvline(x_best[k], color="black", ls=":", alpha=0.6)
        ax.scatter([x_best[k]], [y_best], s=50, color="black",
                   zorder=5, label="x_best")
        ax.set_xlabel(dim_labels[k])
        ax.set_ylabel("f")
        ax.set_title(f"Slice along {dim_labels[k]}")
        ax.legend(fontsize=7)

    # --- extra panel: slice along the line x_true_min -> x_best ---
    if x_true_min is not None:
        ax  = axes[(d) // ncols][(d) % ncols]
        ts  = np.linspace(0.0, 1.0, n_grid)
        X_line = x_true_min + ts[:, None] * (x_best - x_true_min)
        dist   = np.linalg.norm(x_best - x_true_min) * ts

        mu, sigma = gp.predict(X_line, return_std=True)

        ax.fill_between(dist, mu - 2 * sigma, mu + 2 * sigma,
                        alpha=0.15, color="steelblue", label=r"GP $\mu \pm 2 \sigma$")
        ax.fill_between(dist, mu - sigma, mu + sigma,
                        alpha=0.35, color="steelblue", label=r"GP $\mu \pm 1 \sigma$")
        ax.plot(dist, mu, color="steelblue", label="GP mean")

        if objective is not None:
            y_line = np.array([call_objective(row) for row in X_line])
            ax.plot(dist, y_line, color="tomato", ls="--", label="true f")
            y_true_min = call_objective(x_true_min)
            ax.scatter([0.0], [y_true_min], s=60, color="coral",
                       marker="^", zorder=5, label="True Min")

        d_end = float(np.linalg.norm(x_best - x_true_min))
        ax.scatter([d_end], [y_best], s=60, color="black",
                   marker="o", zorder=5, label="BO Best")

        ax.set_xlabel("Distance along line")
        ax.set_ylabel("f")
        ax.set_title("Slice: x_true_min → x_best")
        ax.legend(fontsize=7)

    # Hide any unused subplots
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    return fig

def plot_corner(
    result: dict,
    bounds: list[tuple[float, float]],
    dim_labels: list[str] | None = None,
    figsize: tuple | None = None,
    x_true_min: np.ndarray | None = None,
    bins: int = 20,
    point_alpha: float = 1.0,
    point_size: float = 12.0,
    cmap: str = "viridis",
    color_by_order: bool = True,
) -> plt.Figure:
    """
    Corner plot showing all evaluated points X from a Bayesian optimisation run.

    The lower-triangular off-diagonal panels show 2-D scatter plots of every
    pair of dimensions.  The diagonal panels show 1-D histograms.  Points are
    optionally coloured by evaluation order (earliest = light, latest = dark)
    so that the exploration-to-exploitation transition is visible.

    Special points
    --------------
    x_best     : always shown (black star, ★)
    x_true_min : shown when provided (red triangle, ▲)

    Parameters
    ----------
    result      : dict returned by bayesian_minimization()
    bounds      : list of (lower, upper) per dimension
    dim_labels  : axis labels; defaults to ["x0", "x1", …]
    figsize     : figure size; defaults to 3 × d inches square
    x_true_min  : optional true minimiser to overlay
    bins        : number of histogram bins on diagonal panels
    point_alpha : alpha for the scatter points
    point_size  : marker size for the scatter points
    cmap        : matplotlib colormap used to colour points by order
    color_by_order : if True, colour points by evaluation order;
                     if False, use a single neutral colour

    Returns
    -------
    matplotlib Figure
    """
    X      = result["X"]                                   # (n, d)
    x_best = np.asarray(result["x_best"], dtype=float)    # (d,)
    y_best = result["y_best"]

    bounds_arr = np.asarray(bounds, dtype=float)           # (d, 2)
    d          = bounds_arr.shape[0]
    n          = X.shape[0]

    x_true_min = (
        np.asarray(x_true_min, dtype=float) if x_true_min is not None else None
    )

    if dim_labels is None:
        dim_labels = [f"x{k}" for k in range(d)]

    # ------------------------------------------------------------------ #
    # Figure layout: d × d grid; upper triangle hidden                   #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(
        d, d,
        figsize=figsize or (3.0 * d, 3.0 * d),
        constrained_layout=True,
        squeeze=False,
    )

    # Colour array for scatter (by evaluation order, 0 = first, n-1 = last)
    if color_by_order:
        cmap_obj   = plt.get_cmap(cmap)
        norm_order = np.arange(n) / max(n - 1, 1)          # 0 … 1
        colours    = cmap_obj(norm_order)                   # (n, 4) RGBA
    else:
        colours = np.full((n, 4), [0.3, 0.5, 0.8, point_alpha])

    # ------------------------------------------------------------------ #
    # Helper: unified axis limits with a small margin                     #
    # ------------------------------------------------------------------ #
    def _lim(k: int) -> tuple[float, float]:
        lo, hi = bounds_arr[k]
        margin = 0.04 * (hi - lo)
        return lo - margin, hi + margin

    # ------------------------------------------------------------------ #
    # Fill panels                                                         #
    # ------------------------------------------------------------------ #
    for row in range(d):
        for col in range(d):
            ax = axes[row][col]

            # ---- upper triangle: hide completely ---- #
            if col > row:
                ax.set_visible(False)
                continue

            # ---- diagonal: 1-D histogram ---- #
            if col == row:
                ax.hist(
                    X[:, row],
                    bins=bins,
                    range=(_lim(row)[0], _lim(row)[1]),
                    color="steelblue",
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.4,
                )
                # Vertical lines for special points
                ax.axvline(
                    x_best[row],
                    color="black",
                    lw=1.8,
                    ls="--",
                    label="x_best" if row == 0 else None,
                )
                if x_true_min is not None:
                    ax.axvline(
                        x_true_min[row],
                        color="red",
                        lw=1.8,
                        ls=":",
                        label="x_true_min" if row == 0 else None,
                    )
                ax.set_xlim(_lim(row))
                ax.set_ylabel("count")

            # ---- lower triangle: 2-D scatter ---- #
            else:
                # col < row  →  x-axis = col dimension, y-axis = row dimension
                sc = ax.scatter(
                    X[:, col],
                    X[:, row],
                    c=colours,
                    s=point_size,
                    alpha=point_alpha,
                    linewidths=0,
                    zorder=3,
                )

                # x_best
                ax.scatter(
                    x_best[col],
                    x_best[row],
                    s=5*point_size,
                    marker="*",
                    color="black",
                    zorder=6,
                    label="x_best" if (row == 1 and col == 0) else None,
                )

                # x_true_min
                if x_true_min is not None:
                    ax.scatter(
                        x_true_min[col],
                        x_true_min[row],
                        s=4*point_size,
                        marker="^",
                        color="red",
                        zorder=6,
                        label="x_true_min" if (row == 1 and col == 0) else None,
                    )

                ax.set_xlim(_lim(col))
                ax.set_ylim(_lim(row))

            # ---- axis labels on the border panels only ---- #
            if row == d - 1:
                ax.set_xlabel(dim_labels[col])
            else:
                ax.set_xticklabels([])

            if col == 0 and (col != row):
                ax.set_ylabel(dim_labels[row])
            else:
                ax.set_yticklabels([])

    # ------------------------------------------------------------------ #
    # Colour-bar (evaluation order)                                       #
    # ------------------------------------------------------------------ #
    if color_by_order and d > 1:
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=1, vmax=n),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30)
        cbar.set_label("Evaluation order", fontsize=10)

    # ------------------------------------------------------------------ #
    # Legend                                                             #
    # ------------------------------------------------------------------ #
    fig.legend(fontsize=8, loc="upper right", framealpha=0.7)

    return fig
