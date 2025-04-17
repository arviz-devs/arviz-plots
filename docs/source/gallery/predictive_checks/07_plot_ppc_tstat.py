"""
# PPC with a test statistic

T-statistic for the observed data and posterior predictive data.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_tstat`

EABM chapter on [Posterior predictive checks with summary statistics](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#using-summary-statistics)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("regression1d")
pc = azp.plot_ppc_tstat(
    dt,
    t_stat="median",
    backend="none"
)
pc.show()
