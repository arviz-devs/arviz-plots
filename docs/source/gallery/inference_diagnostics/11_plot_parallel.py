"""
# Parallel coordinates plot

Plot parallel coordinates plot showing posterior points with divergences..

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_parallel`
:::
"""
import numpy as np
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

dt = load_arviz_data("centered_eight")
pc = azp.plot_parallel(
    dt,
    var_name=["theta","tau","mu"],
    norm_method="rank",
    label_type="vert",
    backend="none", # change to preferred backend
)
pc.show()
