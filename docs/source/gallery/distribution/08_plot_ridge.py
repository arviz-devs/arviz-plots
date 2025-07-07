"""
# Ridge plot

Visual representation of marginal distributions over the y axis for a single model

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ridge`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_ridge(
    data,
    backend="none" # change to preferred backend
)
pc.show()