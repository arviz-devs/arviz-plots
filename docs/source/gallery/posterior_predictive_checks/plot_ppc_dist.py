"""
# Kernel Density Plots

Plot of samples from the posterior predictive and observed data.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ppc_dist`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("rugby")
pc = azp.plot_ppc_dist(
    dt,
    pc_kwargs={"aes": {"color": ["__variable__"]}}, # map variable to color
    aes_map={"title": ["color"]}, # change title's color per variable
    backend="none",
)
pc.show()
