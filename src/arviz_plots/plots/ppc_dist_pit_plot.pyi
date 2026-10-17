# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import arviz_base
import xarray as xr
from _typeshed import Incomplete
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument, validate_or_use_rcparam
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.ecdf_plot import plot_ecdf_pit
from arviz_plots.plots.utils import filter_aes, get_visual_kwargs, set_grid_layout
from arviz_plots.plots.utils_ppc import (
    get_ppc_pit,
    get_suspicious_mask_ds,
    prepare_ppc_dist_data,
)
from arviz_plots.visuals import trace_rug

from .plot_collection import PlotCollection

def plot_ppc_dist_pit(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    kind: Literal["auto", "kde", "hist", "ecdf", "dot"] | None = ...,
    num_samples: int = ...,
    method: Literal["pot_c", "prit_c", "piet_c", "envelope"] = ...,
    envelope_prob: float | None = ...,
    coverage: bool = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "predictive_dist",
            "observed_dist",
            "ecdf_lines",
            "credible_interval",
            "suspicious_points",
            "p_value_text",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "predictive_dist",
            "observed_dist",
            "ecdf_lines",
            "credible_interval",
            "suspicious_points",
            "p_value_text",
            "xlabel_dist",
            "xlabel_pit",
            "ylabel",
            "title",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[
        Literal["predictive_dist", "observed_dist", "ecdf_pit"],
        Mapping[str, Any] | xr.Dataset,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
