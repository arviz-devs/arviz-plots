# File generated with docstub

import warnings
from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import arviz_stats
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    ecdf_line,
    fill_between_y,
    hist,
    labelled_title,
    line_x,
    line_xy,
    point_estimate_text,
    remove_axis,
    scatter_x,
    scatter_xy,
    step_hist,
)

from .plot_collection import PlotCollection

def plot_dist(
    dt: DataTree | dict[str, DataTree],
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    kind: Literal["kde", "hist", "dot", "ecdf"] | None = ...,
    point_estimate: Literal["mean", "median", "mode"] | None = ...,
    ci_kind: Literal["eti", "hdi"] | None = ...,
    ci_prob: float | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "title",
            "rug",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "dist",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "title",
            "rug",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[
        Literal["dist", "credible_interval", "point_estimate"],
        Mapping[str, Any] | xr.Dataset,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
