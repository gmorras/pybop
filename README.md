# pybop

Bayesian minimization in arbitrary dimensions using only NumPy and SciPy, with the ability to handle periodic dimensions

## Installation

Clone the repository and install locally:

```bash
pip install .
```

## Dependencies

Core optimization:
- `numpy`
- `scipy`

Plotting:
- `matplotlib`

## Package structure

- `pybop.optimization` — Gaussian process model and Bayesian minimization loop
- `pybop.plotting` — diagnostic plotting helpers

## Quick example

```python
import numpy as np
from pybop import bayesian_minimization, plot_slices

# Define function to minimize (shifted parabola plus cosines)
x_true_min = [2.0, 0.5, 1.2]
def objective(x):
    return (x[0] - x_true_min[0])**2 + (1 - np.cos(x[1] - x_true_min[1])) + (1 - np.cos(2*(x[2] - x_true_min[2])))

# Specify bounds to explore, the dimensions that are periodic and their periods
bounds        = [(-5, 5), (0, 2*np.pi), (0, np.pi)]
periodic_dims = [1, 2]
periods       = [2*np.pi, np.pi]

# Run bayesian minimization
result = bayesian_minimization(
    objective, bounds,
    periodic_dims=periodic_dims, periods=periods,
    n_initial=8, n_calls=30,
    verbose=True,
)

# Print result
print("Best x:", result["x_best"])
print("Best y:", result["y_best"])
```

The returned `result` dictionary contains:

- `x_best`: best point found
- `y_best`: best objective value found
- `X`: all evaluated points
- `y`: all objective values
- `gp`: final fitted Gaussian process

## Plotting example

```python
import matplotlib.pyplot as plt
from pybop import plot_slices, plot_corner

fig1 = plot_slices(result, bounds, objective=objective, x_true_min=x_true_min)
fig2 = plot_corner(result, bounds, x_true_min=x_true_min)

plt.show()
```

## Feedback & Issues

We welcome feedback, bug reports, and feature requests! If you encounter any issues or have suggestions for improvements, please open an issue in the GitHub issue tracker.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

