"""
# 2D histogram of all variables against each other

Use a two-dimensional histogram when a scatter plot is too dense.

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
    visuals={"scatter": False, "histogram2d": True},
    stats={"histogram2d": {"bins": 20}},
    marginal=True,
    backend="none",  # change to preferred backend
)
pc.show()
