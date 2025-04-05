"""
# PAV-adjusted calibration

PAV-adjusted calibration plot for binary predictions.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_pava`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("anes")
pc = azp.plot_ppc_pava(
    dt,
    backend="none",
)
pc.show()
