"""
# Rootogram

Rootogram for the posterior predictive and observed data.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_rootogram`

EABM chapter on [Posterior predictive checks for count data](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#posterior-predictive-checks-for-count-data)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("rugby")
pc = azp.plot_ppc_rootogram(
    dt,
    aes={"color": ["__variable__"]}, # map variable to color
    aes_by_visuals={"title": ["color"]}, # change title's color per variable
    figure_title="Posterior Predictive Rootogram for Rugby Model",
    backend="none",
)
pc.show()
