"""
(gallery_dist_models)=
# Marginal distribution comparison plot

Full marginal distribution comparison between different models

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_dist`

Other gallery examples using `plot_dist`: {ref}`gallery_dist_kde`, {ref}`gallery_dist_ecdf`

Other examples comparing marginal distributions: {ref}`gallery_forest_models`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

c = load_arviz_data("centered_eight")
n = load_arviz_data("non_centered_eight")
pc = azp.plot_dist(
    {"Centered": c, "Non Centered": n},
    backend="none"  # change to preferred backend
)
pc.show()
