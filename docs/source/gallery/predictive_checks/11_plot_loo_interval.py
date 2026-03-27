"""
# Interval plot
Plot LOO posterior predictive point estimate and intervals at each observation.
---
:::{seealso}
API Documentation: {func}`~arviz_plots.plot_loo_interval`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("radon")

data_subset = data.isel(obs_id=range(50))

pc = azp.plot_loo_interval(
    data_subset,
    backend="none",
)

pc.show()