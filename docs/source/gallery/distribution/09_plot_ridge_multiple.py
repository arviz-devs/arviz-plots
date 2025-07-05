"""
# Ridge plot for multiple models

Visual representation of marginal distributions over the y axis showing centered and non-centered schools

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_ridge`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

centered = load_arviz_data("centered_eight")
non_centered = load_arviz_data("non_centered_eight")

pc = azp.plot_ridge(
    {
        "centered": centered,
        "non-centered": non_centered
    },
    backend="none"
)
pc.add_legend("model")
pc.show()