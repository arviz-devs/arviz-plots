"""
# Add Reference Lines

Draw reference lines on plots to highlight specific thresholds, targets, or important values.
"""

import xarray as xr
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
post = data.posterior.dataset
ref_ds = xr.concat([post.mean(), post.min(), post.max()], dim="ref_line_dim")
pc = azp.plot_dist(
    data,
    kind="ecdf",
    backend="none",   # change to preferred backend
)
pc = azp.add_reference_lines(pc, references=ref_ds, aes_map={"ref_line": ["color", "linestyle"]})
pc.show()
