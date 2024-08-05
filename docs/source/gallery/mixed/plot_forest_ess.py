"""
# Forest plot with ESS

Multiple panel visualization with a forest plot and ESS information

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_forest`
:::
"""
from importlib import import_module

import arviz_stats  # make azstats accessor available
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

backend="none"  # change to preferred backend
plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
color = plot_bknd.get_default_aes("color", 1, {})[0]

centered = load_arviz_data("centered_eight")
c_aux = centered["posterior"].expand_dims(
    column=3
).assign_coords(column=["labels", "forest", "ess"])
pc = azp.plot_forest(c_aux, combined=True, backend=backend)
pc.map(
    azp.visuals.scatter_x, "ess",
    data=centered.posterior.ds.azstats.ess(),
    coords={"column": "ess"}, color=color
)
pc.show()
