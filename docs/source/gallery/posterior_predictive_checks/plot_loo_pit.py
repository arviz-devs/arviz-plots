"""
# LOO-PIT ECDF

Plot of $p(\tilde y_i \le y_i \mid y_{-i})$, where $y_i$ represents the observed data for index $i$
and $\tilde y_i$ represents the posterior predictive sample at index $i$. $y_{-i}$ indicates we have
left out the $i$-th observation.


The distribution should be uniform if the model is well-calibrated. As small deviations from
uniformity are expected, the plot also shows the credible bands. 
---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_pit`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("radon")
pc = azp.plot_loo_pit(
    dt,
    backend="none",
)
pc.show()
