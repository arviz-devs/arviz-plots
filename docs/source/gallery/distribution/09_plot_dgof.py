"""
# Diagnostics for density estimation

Diagnostics for assessing the goodness-of-fit of estimated distributions
to the underlying data using the Probability Integral Transform (PIT) and
the Δ-ECDF-PIT plots.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_dgof`

EABM chapter on [Visualization of Random Variables with ArviZ](https://arviz-devs.github.io/EABM/Chapters/Distributions.html)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("radon")
pc = azp.plot_dgof(dt,
    var_names="g",
    kind="hist",
    stats={"dist": {"bins":30}},
    backend="none"  # change to preferred backend
)
pc.show()
