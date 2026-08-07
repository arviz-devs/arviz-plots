# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import numpy as np
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
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.rank_plot import plot_rank
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
    set_grid_layout,
)
from arviz_plots.visuals import labelled_x, labelled_y, ticklabel_props, trace_rug

from .plot_collection import PlotCollection

def plot_rank_dist(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: Incomplete = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    compact: bool = ...,
    combined: bool = ...,
    kind: Literal["kde", "hist", "ecdf", "dot"] | None = ...,
    envelope_prob: float | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal["dist", "rank", "label", "ticklabels", "xlabel_rank"], Sequence[str]
    ] = ...,
    visuals: Mapping[
        Literal[
            "dist",
            "rank",
            "label",
            "ticklabels",
            "xlabel_rank",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["dist", "ecdf_pit"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
