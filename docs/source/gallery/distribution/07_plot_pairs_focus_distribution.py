"""
# Pairwise focus distribution without divergence

Pairwise focus distribution of a variable against all other variables in the dataset.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_pairs_focus`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_pairs_focus(
    data,
    var_names=["theta","tau"],
    focus_var="mu",
    backend="none", # change to preferred backend
    visuals = {"divergence": False},
)
pc.show()
