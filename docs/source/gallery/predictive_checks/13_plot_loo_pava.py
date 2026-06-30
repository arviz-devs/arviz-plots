"""
# LOO PAV-adjusted calibration

PAV-adjusted calibration plot using leave-one-out (LOO) cross-validation to resample
the posterior predictive distribution. Ideal for binary, ordinal or categorical data.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_loo_pava`

EABM chapter on [Posterior predictive checks for binary data](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#posterior-predictive-checks-for-binary-data)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("anes")
pc = azp.plot_loo_pava(
    dt,
    backend="none",
)
pc.show()
