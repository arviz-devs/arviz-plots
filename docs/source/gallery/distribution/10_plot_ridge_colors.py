"""
# Color-mapped ridge plot

A single model ridge plot with colors mapped to the variables

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ridge`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")
non_centered = load_arviz_data("non_centered_eight")

pc = azp.plot_ridge(
    non_centered,
    var_names=["theta", "mu", "theta_t", "tau"],
    aes={"color": ["__variable__"]},
    figure_kwargs={"width_ratios":[1,2]},
    aes_by_visuals={"labels": ["color"]},
    shade_label="school",
    backend="none"
)
pc.show()
