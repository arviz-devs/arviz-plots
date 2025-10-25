"""
# Pareto shape parameter diagnostics

Visualize Pareto k diagnostics from PSIS-LOO-CV to assess the reliability of
importance sampling for each observation.

The Pareto k diagnostic indicates how reliable the importance sampling approximation
is for each observation. Values below 0.7 are generally considered good, while higher
values suggest the importance weights are unreliable and the LOO estimates may be
inaccurate for those observations.

This plot helps identify problematic observations that may be influential causing
the importance sampling to be unreliable.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_khat`
:::
"""
from arviz_base import load_arviz_data
from arviz_stats import loo

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("radon")
elpd_data = loo(dt, pointwise=True)

pc = azp.plot_khat(
    elpd_data,
    threshold=0.7,
    show_hlines=True,
    show_bins=True,
    backend="none",  # change to preferred backend
)

pc.show()
