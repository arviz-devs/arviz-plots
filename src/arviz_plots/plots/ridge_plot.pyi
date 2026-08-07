# File generated with docstub

import warnings
from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import arviz_stats
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection, process_facet_dims
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
)
from arviz_plots.visuals import (
    annotate_label,
    fill_between_y,
    hist,
    line_xy,
    remove_axis,
    scatter_xy,
)

from .plot_collection import PlotCollection

def plot_ridge(
    dt: DataTree | dict[str, DataTree],
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    combined: bool = ...,
    ridge_height: float = ...,
    labels: Sequence[str] | None = ...,
    shade_label: str | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    kind: Literal["kde", "ecdf", "hist", "dot"] | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "edge",
            "face",
            "labels",
            "shade",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "edge",
            "face",
            "labels",
            "shade",
            "ticklabels",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Mapping,
) -> PlotCollection: ...
