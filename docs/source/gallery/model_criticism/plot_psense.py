"""
# Power scaling prior sensitivity plot

Plot of power scaling prior sensitivity distribution

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_psense_dist`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

idata = load_arviz_data("rugby")
pc = azp.plot_psense_dist(
    idata,
    var_names=["intercept", "home", "defs"],
    coords={"team": ["Scotland", "Wales"]},
    backend="none",
)
pc.show()
