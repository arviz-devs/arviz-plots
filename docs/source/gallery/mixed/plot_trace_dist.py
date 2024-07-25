"""
(gallery_trace_dist)=
# Trace and distribution plot

Two column layout with marginal distributions on the left and MCMC traces on the right

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_trace_dist`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

data = load_arviz_data("non_centered_eight")
pc = azp.plot_trace_dist(
    data,
    backend="none"  # change to preferred backend
)
pc.show()
