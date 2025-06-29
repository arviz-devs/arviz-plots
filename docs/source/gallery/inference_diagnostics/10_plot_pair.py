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
    visuals={"divergence":True},
    marginal=False,
    triangle="both",
    backend="none", # change to preferred backend
)
pc.show()
