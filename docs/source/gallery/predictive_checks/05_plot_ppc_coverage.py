"""
# Coverage ECDF

Proportion of true values that fall within a given prediction interval.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_pit`

EABM chapter on [Posterior predictive checks with summary statistics](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#coverage)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("radon")
pc = azp.plot_ppc_pit(
    dt,
    coverage=True,
    backend="none",
)
pc.show()
