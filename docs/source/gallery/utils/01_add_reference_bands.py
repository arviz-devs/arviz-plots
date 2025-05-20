"""
# Add Reference Bands

Draw reference bands to highlight specific regions.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.add_bands`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
rope = [(-1, 1)]
pc = azp.plot_forest(
    data,
    backend="none",   # change to preferred backend
)

pc = azp.add_bands(
    pc,
    values=rope,
    aes_map={"ref_band": ["color"]},
    color=["#f66d7f"]
)

pc.show()
