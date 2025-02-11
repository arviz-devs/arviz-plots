"""
# PAV-adjusted calibration plot

Plot of the PAV-adjusted calibration for binary classifier.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_psense_dist`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

idata = load_arviz_data("classification10d")
pc = azp.plot_pava_calibration(
    idata,
    backend="none",
)
pc.show()
