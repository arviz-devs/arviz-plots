"""
# Convergence diagnostics distribution plot

Plot the distribution of ESS and R-hat.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_convergence_dist`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("radon")
pc = azp.plot_convergence_dist(
    data,
    var_names=["za_county"],
    backend="none",  # change to preferred backend
)

pc.show()
