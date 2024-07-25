"""
(gallery_forest)=
# Forest plot

Default forest plot with marginal distribution summaries

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_forest`

Other gallery examples using `plot_forest`: {ref}`gallery_forest_shade`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

data = load_arviz_data("rugby")
pc = azp.plot_forest(
    data,
    var_names=["home", "atts", "defs"],
    backend="none"  # change to preferred backend
)
pc.show()
