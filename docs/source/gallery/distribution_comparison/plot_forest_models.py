"""
(gallery_forest_models)=
# Forest plot comparison

Forest plot summaries for 1D marginal distributions

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_forest`

Other gallery examples using `plot_forest`: {ref}`gallery_forest`, {ref}`gallery_forest_shade`

Other examples comparing marginal distributions: {ref}`gallery_dist_models`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

c = load_arviz_data("centered_eight")
n = load_arviz_data("non_centered_eight")
pc = azp.plot_forest(
    {"Centered": c, "Non Centered": n},
    backend="none"  # change to preferred backend
)
pc.show()
