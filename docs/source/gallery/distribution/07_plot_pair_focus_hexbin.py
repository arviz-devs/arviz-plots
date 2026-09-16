"""
# Hexagonal histogram of one variable against other variables

Use hexagonal bins to show dense relationships between one variable and other variables in the
dataset. Statistical options, such as the grid size, are set through ``stats``.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_pair_focus`
:::
"""

import numpy as np
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
data.posterior["log_tau"] = np.log(data.posterior["tau"])

pc = azp.plot_pair_focus(
    data,
    var_names=["theta"],
    focus_var="log_tau",
    visuals={"scatter": False, "hexbin": True},
    stats={"hexbin": {"gridsize": 20}},
    backend="none",  # change to preferred backend
)
pc.show()
