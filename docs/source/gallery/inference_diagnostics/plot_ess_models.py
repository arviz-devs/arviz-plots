"""
# ESS comparison plot

Full ESS (Either local or quantile) comparison between different models

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ess`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

c = load_arviz_data("centered_eight")
n = load_arviz_data("non_centered_eight")
pc = azp.plot_ess(
    {"Centered": c, "Non Centered": n},
    backend="none",  # change to preferred backend
)
pc.show()
