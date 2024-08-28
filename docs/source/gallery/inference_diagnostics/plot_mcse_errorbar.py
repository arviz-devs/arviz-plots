"""
# MCSE Quantile plot with errorbars

Facetted quantile MCSE plot with errorbars

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_mcse`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

data = load_arviz_data("centered_eight")
pc = azp.plot_mcse(
    data,
    errorbar=True,
    backend="none",  # change to preferred backend
)
pc.show()
