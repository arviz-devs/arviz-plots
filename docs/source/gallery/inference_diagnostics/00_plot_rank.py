"""
# Rank plot

faceted plot with fractional ranks for each variable

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_rank`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_rank(
    data,
    backend="none"  # change to preferred backend
)
pc.show()
