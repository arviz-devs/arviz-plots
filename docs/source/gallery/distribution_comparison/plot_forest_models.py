"""
# Forest plot comparison

Forest plot summaries for 1D marginal distributions

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_forest`

Other examples comparing marginal distributions: {ref}`gallery_dist_models`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

c = load_arviz_data("centered_eight")
n = load_arviz_data("non_centered_eight")
pc = azp.plot_forest(
    {"Centered": c, "Non Centered": n},
    backend="none"  # change to preferred backend
)
pc.show()
