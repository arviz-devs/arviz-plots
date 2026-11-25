"""
# Linear model plot

Posterior predictive and mean plots for regression-like data. The `plot_lm` function
visualizes credible intervals around predictions alongside observed data points.

The examples below demonstrate `plot_lm` with simulated data and the
`roaches_zinb` dataset, including cases where the prediction variable and
observed response variable have different names.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_lm`

:::
"""

import numpy as np
from arviz_base import from_dict, load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

# Example 1: Linear model with simulated data

np.random.seed(42)
x_data = np.random.normal(0, 1, 100)
y_data = 2 + x_data * 0.5 + np.random.normal(0, 0.5, 100)
y_data_rep = np.random.normal(2 + x_data * 0.5, 0.5, (4, 200, 100))

dt = from_dict(
    {
        "posterior_predictive": {"y": y_data_rep},
        "observed_data": {"y": y_data},
        "constant_data": {"x": x_data},
    },
    dims={"y": ["obs_id"], "x": ["obs_id"]},
    coords={"obs_id": range(100)},
)

pc = azp.plot_lm(
    dt,
    backend="matplotlib",  # change to preferred backend
)
pc.show()


# Example 2: Linear model with the roaches_zinb dataset

dt_roaches = load_arviz_data("roaches_zinb")

# Plot the observed binary response using posterior predictive samples.
pc = azp.plot_lm(
    dt_roaches,
    x="roach count",
    y="y_pos",
    y_obs="y_pos",
    backend="matplotlib",
)
pc.show()


# Example 3: Plot the posterior mean parameter.
# Here, the prediction variable (mu) and observed response variable (y)
# have different names.

pc = azp.plot_lm(
    dt_roaches,
    x="roach count",
    y="mu",
    y_obs="y",
    group="posterior",
    visuals={"observed_scatter": False},
    backend="matplotlib",
)
pc.show()


# Example 4: Plot the posterior probability parameter.
# The prediction variable (psi) is matched with the observed binary response (y_pos).

pc = azp.plot_lm(
    dt_roaches,
    x="roach count",
    y="psi",
    y_obs="y_pos",
    group="posterior",
    visuals={"observed_scatter": False},
    backend="matplotlib",
)
pc.show()
