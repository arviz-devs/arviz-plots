"""
Posterior Predictive Intervals Plot
===================================

Plot posterior predictive intervals for each observation against the observed data.
This plot helps to check if the model's predictions are well-calibrated and
if they manage to capture the true observed values.

---
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("radon")

data_subset = data.isel(obs_id=range(50))

pc = azp.plot_ppc_intervals(
    data_subset,
    var_names=["y"],
    backend="none",
)

pc.show()