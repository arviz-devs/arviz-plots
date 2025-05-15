"""
# Add Reference Bands

Draw reference bands on plots to highlight specific regions.

---

:::{seealso}
API Documentation: {func}`~arviz_plots.add_reference_bands`
:::
"""

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
ref_dict = [(0, 3), (3, 9), (10, 20)]
pc = azp.plot_dist(
    data,
    kind="ecdf",
    backend="none",   # change to preferred backend
)

pc = azp.add_reference_bands(
    pc,
    references=ref_dict,
    aes_map={"ref_band": ["color"]},
    color=["black", "gray", "blue"]
)

pc.show()
