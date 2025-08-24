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

styling = {
    "outer_interval": {"width": 1.0, "color": "C0", "alpha": 0.5},
    "inner_interval": {"width": 2.5, "color": "C0"},
    "point_estimate": {"s": 20, "color": "C0"},
    "observed": {"s": 25, "marker": "o", "edgecolor": "k", "facecolor": "none"}
}

pc = azp.plot_ppc_intervals(
    data_subset,
    var_names=["y"],
    visuals=styling,
    backend="none"
)

pc.show()