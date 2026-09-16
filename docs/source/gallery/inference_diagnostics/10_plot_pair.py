"""
# Scatter plot of all variables against each other with divergences

Plot all variables against each other in the dataset.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_pair`
:::
"""

import numpy as np
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("centered_eight")
pc = azp.plot_pair(
    dt,
    var_names=["theta", "tau"],
    coords={"school": ["Lawrenceville", "Mt. Hermon"]},
    visuals={"divergence": True},
    marginal=False,
    backend="none",  # change to preferred backend
)
pc.show()

# %%
# Use a two-dimensional histogram when the scatter plot is too dense. Statistical
# options, such as the number of bins, are set through ``stats``.
pc = azp.plot_pair(
    dt,
    var_names=["theta", "tau"],
    coords={"school": ["Lawrenceville", "Mt. Hermon"]},
    visuals={"scatter": False, "histogram2d": True},
    stats={"histogram2d": {"bins": 20}},
    marginal=True,
    backend="none",  # change to preferred backend
)
pc.show()

# %%
# Hexagonal bins are available through the same interface.
pc = azp.plot_pair(
    dt,
    var_names=["theta", "tau"],
    coords={"school": ["Lawrenceville", "Mt. Hermon"]},
    visuals={"scatter": False, "hexbin": True},
    stats={"hexbin": {"gridsize": 25}},
    marginal=True,
    backend="none",  # change to preferred backend
)
pc.show()
