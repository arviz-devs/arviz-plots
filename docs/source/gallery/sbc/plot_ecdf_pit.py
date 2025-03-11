"""
# PIT-ECDF difference

faceted plot with PIT Δ-ECDF values for each variable

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ecdf_pit`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("sbc")
pc = azp.plot_ecdf_pit(
    data,
    backend="none"  # change to preferred backend
)
pc.show()
