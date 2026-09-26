"""
# Hexagonal histograms of all variables against each other

Use hexagonal bins to show dense relationships between variables.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_pair`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_pair(
    data,
    var_names=["theta", "tau"],
    coords={"school": ["Lawrenceville", "Mt. Hermon"]},
    visuals={"scatter": False, "hexbin": True},
    marginal=True,
    backend="none",  # change to preferred backend
)
pc.show()
