"""
# Coverage ECDF

Coverage refers to the proportion of true values that fall within a given prediction interval.
For a well-calibrated model, the coverage should match the intended interval width. For example,
a 95% credible interval should contain the true value 95% of the time.
As a result, the Empirical Cumulative Distribution Function (ECDF) of these intervals should be
uniformly distributed.

To make the plot easier to interpret, we plot the Δ-ECDF, that is, the difference between
the expected CDF from the observed ECDF. As small deviations from uniformity are expected, 
the plot also shows the credible envelope. 

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_pit`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("radon")
pc = azp.plot_ppc_pit(
    dt,
    coverage=True,
    backend="none",
)
pc.show()
