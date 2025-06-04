"""
# Posterior Histograms

Faceted histogram plots for 1D marginals of the distribution.
 The `point_estimate_text` option is set to False to omit that visual from the plot.
---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_dist`

EABM chapter on [Visualization of Random Variables with ArviZ](https://arviz-devs.github.io/EABM/Chapters/Distributions.html#distributions-in-arviz)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_dist(
    data,
    kind="hist",
    visuals={"point_estimate_text": False},
    backend="none"  # change to preferred backend
)
pc.show()
