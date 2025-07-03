"""
# Ridge plot

1D ridge plot with marginal distribution summaries

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ridge`
:::
"""
from arviz_base import load_arviz_data
import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_ridge(data, backend="none")
pc.show()