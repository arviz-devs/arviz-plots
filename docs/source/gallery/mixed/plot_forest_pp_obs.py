"""
(gallery_forest_pp_obs)=
# Posterior predictive and observations forest plot

Overlay of forest plot for the posterior predictive samples and the actual observations

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_forest`

Other gallery examples using `plot_forest`: {ref}`gallery_forest`, {ref}`gallery_forest_shade`
:::
"""
from importlib import import_module

import arviz_stats  # make azstats accessor available
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

centered = load_arviz_data("non_centered_eight")
pc = azp.plot_forest(
    centered,
    group="posterior_predictive",
    combined=True,
    backend="none"  # change to preferred backend
)
pc.map(
    azp.visuals.scatter_x,
    "observations",
    data=centered.observed_data.ds,
    coords={"column": "forest"},
    color="black",
)
pc.show()
