"""
# Energy

Plot transition and marginal energy distributions

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_energy`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_energy(
    data,
    backend="none"  # change to preferred backend
)
pc.show()