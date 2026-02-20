"""
# Linear model plot

Posterior predictive plot for regression-like data. The `plot_lm` function visualizes
credible intervals around predictions alongside observed data points. It expects
three groups in the input DataTree: `constant_data` for the independent variable (x-axis),
`observed_data` for the response variable, and `posterior_predictive` (default) or `posterior`
for the predicted values.

The main visual elements are:

- **pe_line**: Point estimate line (mean, median, or mode of predictions).
- **ci_band**: Filled credible interval band, suited for continuous x variables.
- **ci_vlines**: Vertical credible interval lines, suited for discrete or categorical x variables.
- **ci_bounds**: Credible interval boundary lines (alternative to the filled band).
- **observed_scatter**: Scatter of observed data points.

This example uses synthetic data from a simple linear regression model
y = 2 + 0.5 * x + noise. The posterior predictive samples are generated
by adding Gaussian noise to the true regression line. By default, `plot_lm`
applies Savitzky-Golay smoothing and shows a filled credible interval band
(`ci_band`) together with a point estimate line (`pe_line`) and observed
data scatter (`observed_scatter`).

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_lm`

Old ArviZ `plot_lm` [documentation](https://python.arviz.org/en/v0.23.0/api/generated/arviz.plot_lm.html)
:::
"""
import numpy as np
from arviz_base import from_dict

import arviz_plots as azp

azp.style.use("arviz-variat")

np.random.seed(42)
x_data = np.random.normal(0, 1, 100)
y_data = 2 + x_data * 0.5
y_data_rep = np.random.normal(y_data, 0.5, (4, 200, 100))

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
    backend="none",  # change to preferred backend
)
pc.show()
