"""
# T-Statistic

t-statistic for the posterior/prior predictive data.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_tstat`
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
