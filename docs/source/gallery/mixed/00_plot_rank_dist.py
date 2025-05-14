"""
# Rank and distribution plot

Two column layout with marginal distributions on the left and fractional ranks on the right

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_rank_dist`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("non_centered_eight")
pc = azp.plot_rank_dist(
    data,
    var_names=["mu", "tau"],
    backend="none"  # change to preferred backend
)
pc.show()
