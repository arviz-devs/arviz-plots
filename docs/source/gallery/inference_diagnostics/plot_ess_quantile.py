"""
# ESS Quantile plot

faceted quantile ESS plot

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ess`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_ess(
    data,
    kind="quantile",
    backend="none",  # change to preferred backend
)
pc.show()
