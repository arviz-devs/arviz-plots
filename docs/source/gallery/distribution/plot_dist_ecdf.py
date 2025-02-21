"""
# Posterior ECDFs

faceted ECDF plots for 1D marginals of the distribution

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_dist`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_dist(
    data,
    kind="ecdf",
    backend="none"  # change to preferred backend
)
pc.show()
