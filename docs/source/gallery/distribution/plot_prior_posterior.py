"""
# Plot prior and posterior

Plot prior and posterior marginal distributions.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_prior_posterior`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_prior_posterior(
    data,
    var_names="mu",
    kind="hist",
    backend="none"  # change to preferred backend
)
pc.show()
