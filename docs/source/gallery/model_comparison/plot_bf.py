"""
# Bayes_factor
Compute Bayes factor using Savage–Dickey ratio
---
:::{seealso}
API Documentation: {func}`~arviz_plots.plot_bf`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")

pc = azp.plot_bf(
    data,
    backend="none",
    var_name="mu"
)

pc.show()