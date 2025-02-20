"""
# Rootogram plot

Rootogram for the posterior predictive and observed data.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_rootogram`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("rugby")
pc = azp.plot_ppc_rootogram(
    dt,
    backend="none",
)
pc.show()
