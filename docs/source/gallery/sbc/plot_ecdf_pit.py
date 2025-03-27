"""
# PIT-ECDF difference

Faceted plot with PIT Δ-ECDF values for each variable

The ``plot_ecdf_pit`` function assumes the values passed to it has already been transformed
to PIT values, as in the case of SBC analysis or values from ``arviz_base.loo_pit``.

The distribution should be uniform if the model is well-calibrated. 

To make the plot easier to interpret, we plot the Δ-ECDF, that is, the difference between
the expected CDF from the observed ECDF. As small deviations from uniformity are expected, 
the plot also shows the credible envelope. 

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ecdf_pit`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("sbc")
pc = azp.plot_ecdf_pit(
    data,
    backend="none"  # change to preferred backend
)
pc.show()
