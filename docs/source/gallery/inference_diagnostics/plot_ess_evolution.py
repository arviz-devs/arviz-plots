"""
# ESS Evolution plot

Facetted plot with ESS 'bulk' and 'tail' for each variable

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ess_evolution`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

data = load_arviz_data("centered_eight")
pc = azp.plot_ess_evolution(data, backend="none")  # change to preferred backend
pc.show()
