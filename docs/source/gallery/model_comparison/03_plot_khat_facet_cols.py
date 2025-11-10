"""
# Pareto k diagnostics with column faceting

Faceted Pareto k plot using column layout to compare diagnostics across field dimensions

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_khat`
:::
"""
import numpy as np
from arviz_base import load_arviz_data
from arviz_base.labels import MapLabeller
from arviz_stats import loo

import arviz_plots as azp

azp.style.use("arviz-variat")

idata = load_arviz_data("rugby")
points_da = idata.log_likelihood.dataset[["home_points", "away_points"]].to_array(
    dim="field"
)
idata.log_likelihood["points"] = points_da.assign_coords(field=["home", "away"])
idata.log_likelihood.coords["year"] = ("match", np.repeat([2014, 2015, 2016, 2017], 15))

loo_result = loo(idata, var_name="points", pointwise=True)

pc = azp.plot_khat(
    loo_result,
    cols=["field"],
    labeller=MapLabeller(coord_map={"field": {"home": "Home", "away": "Away"}}),
    visuals={
        "bin_text": True,
        "hlines": {"color": "darkblue", "width": 2},
        "title": True,
    },
    figure_kwargs={"figsize": (6, 4)},
    backend="none",  # change to preferred backend
)

pc.show()
