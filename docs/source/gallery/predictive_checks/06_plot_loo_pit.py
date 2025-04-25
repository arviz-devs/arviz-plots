"""
# LOO-PIT ECDF

Plot of the probability integral transform of the posterior predictive distribution with
respect to the observed data using the leave-one-out (LOO) method.


---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_pit`

EABM chapter on [Posterior predictive checks with summary statistics](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#sec-avoid-double-dipping)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("radon")
pc = azp.plot_loo_pit(
    dt,
    backend="none",
)
pc.show()
