"""
# Survival analysis (censored data)
Plot Kaplan-Meier survival curve vs posterior predictive draws.
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
    extrapolation_factor=None,
    backend="none",
)

pc.show()