"""
# Censored posterior predictive checks
Plot Kaplan-Meier survival curves for posterior predictive checking of censored data.
---
:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_censored`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("censored_cats")
pc = azp.plot_ppc_censored(
    data,
    truncation_factor=None,
    backend="none",
)

pc.show()