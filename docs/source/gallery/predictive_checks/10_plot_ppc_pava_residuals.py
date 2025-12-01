"""
# PAV-adjusted residual plot

Residual plot using PAV-adjusted calibration for binary predictions.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_pava_residuals`

EABM chapter on [Posterior predictive checks for binary data](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#posterior-predictive-checks-for-binary-data)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("roaches_zinb")
pc = azp.plot_ppc_pava_residuals(
    dt,
    var_names="y_pos",
    x_var="roach count",
    backend="none",
)
pc.show()
