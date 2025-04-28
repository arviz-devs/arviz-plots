"""
# Predictive check with KDEs

Plot of samples from the posterior predictive and observed data.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_dist`

EABM chapter on [Posterior predictive checks](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#posterior-predictive-checks)
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
