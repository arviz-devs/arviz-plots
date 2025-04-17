"""
# Sensitivity posterior marginals

The posterior sensitivity is assessed by power-scaling the prior or likelihood and
visualizing the resulting changes. Sensitivity can then be quantified by considering
how much the perturbed posteriors differ from the base posterior.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_psense_dist`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

idata = load_arviz_data("rugby")
pc = azp.plot_psense_dist(
    idata,
    var_names=["defs", "sd_att", "sd_def"],
    coords={"team": ["Scotland", "Wales"]},
    pc_kwargs={"y": [-2, -1, 0]},
    backend="none",
)
pc.show()
