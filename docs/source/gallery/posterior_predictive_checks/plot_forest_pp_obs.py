"""
# Posterior predictive and observations forest plot

Overlay of forest plot for the posterior predictive samples and the actual observations

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_forest`
:::
"""
from importlib import import_module

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

backend="none"  # change to preferred backend
plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
color = plot_bknd.get_default_aes("color", 3, {})[-1]

idata = load_arviz_data("non_centered_eight")
pc = azp.plot_forest(
    idata,
    group="posterior_predictive",
    combined=True,
    labels=["obs_dim_0"],
    backend=backend,
)
pc.map(
    azp.visuals.scatter_x,
    "observations",
    data=idata.observed_data.ds,
    coords={"column": "forest"},
    color=color,
)
target = pc.viz["plot"].sel(column="forest").item()
plot_bknd.xlabel("Observations", target)
pc.show()
