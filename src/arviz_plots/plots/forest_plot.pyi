# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_stats
import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from _typeshed import Incomplete as labeller
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from numpy.typing import ArrayLike

from arviz_plots.plot_collection import PlotCollection, process_facet_dims
from arviz_plots.plots.utils import filter_aes, get_visual_kwargs, process_group_variables_coords
from arviz_plots.visuals import annotate_label, fill_between_y, line_x, remove_axis, scatter_x

def plot_forest(
    dt: xarray.DataTree | dict[str, xarray.DataTree],
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    combined: bool = ...,
    point_estimate: Literal["mean", "median", "mode"] | None = ...,
    ci_kind: Literal["eti", "hdi"] | None = ...,
    ci_probs: ArrayLike | None = ...,
    labels: Sequence[str] | None = ...,
    shade_label: str | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh"] | None = ...,
    labeller: labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "credible_interval",
            "point_estimate",
            "labels",
            "shade",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "trunk",
            "twig",
            "point_estimate",
            "labels",
            "shade",
            "ticklabels",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[
        Literal["trunk", "twig", "point_estimate"], Mapping[str, Any] | xr.Dataset
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
