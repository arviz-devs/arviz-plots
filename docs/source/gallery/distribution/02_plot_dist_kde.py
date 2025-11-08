"""
# Posterior KDEs

KDE plot of the variable `mu` from the centered eight model. The `sample_dims` parameter is
used to restrict the KDE computation along the `draw` dimension only.
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
    kind="kde",
    var_names=["mu"],
    sample_dims=["draw"],    
    backend="none"  # change to preferred backend
)
pc.show()
