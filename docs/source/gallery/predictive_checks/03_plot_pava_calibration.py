"""
# PAV-adjusted calibration

PAV-adjusted calibration plot for binary predictions.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_pava`

EABM chapter on [Posterior predictive checks for binary data](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#posterior-predictive-checks-for-binary-data)
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
