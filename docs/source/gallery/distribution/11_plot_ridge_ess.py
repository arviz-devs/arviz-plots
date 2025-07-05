"""
# Single-model ridge plot with ess estimations

Visual representation of marginal distributions over the y axis with additional ess estimations

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ridge`
:::
"""
import arviz_stats
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")
centered = load_arviz_data("centered_eight")

c_aux = centered["posterior"].dataset.expand_dims(column=3).assign_coords(column=["labels", "ridge", "ess"])

pc = azp.plot_ridge(
    c_aux,
    combined=True,
    figure_kwargs={"figsize_units": "dots", "figsize": [570, 450]},
    backend="none"
)
pc.map(
    azp.visuals.scatter_x, "ess",
    data=centered.azstats.ess().ds,
    coords={"column": "ess"}, color="crimson"
)
pc.show()
