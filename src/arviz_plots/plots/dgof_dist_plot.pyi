# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dgof_plot import plot_dgof
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    get_visual_kwargs,
    process_group_variables_coords,
    set_grid_layout,
)

from .plot_collection import PlotCollection

def plot_dgof_dist(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    kind: Literal["kde", "hist", "dot"] | None = ...,
    method: Literal["pot_c", "prit_c", "piet_c", "envelope"] = ...,
    envelope_prob: float | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal["dist", "ecdf_lines", "credible_interval", "title", "xlabel", "ylabel"],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "dist",
            "ecdf_lines",
            "credible_interval",
            "title",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["dist", "ecdf_pit"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
