"""
# Posterior Predictive Checks

Plot of samples from the posterior predictive and observed data.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_pava_calibration`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("radon")
pc = azp.plot_ppc_dist(
    dt,
    backend="none",
)
pc.show()
