"""
# Pareto shape parameter diagnostics

Default Pareto k diagnostic plot from PSIS-LOO-CV to assess importance sampling reliability

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_khat`
:::
"""
from arviz_base import load_arviz_data
from arviz_stats import loo

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("rugby")
elpd_data = loo(dt, var_name="home_points", pointwise=True)

pc = azp.plot_khat(
    elpd_data,
    threshold=0.7,
    visuals={"hlines": True, "bin_text": True},
    backend="none",
)

pc.show()
