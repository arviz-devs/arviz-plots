# File generated with docstub

import warnings
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
from arviz_stats.ecdf_utils import ecdf_pit
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    annotate_xy,
    ecdf_line,
    fill_between_y,
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    remove_axis,
    scatter_xy,
    set_xticks,
    set_ylim,
)

from .plot_collection import PlotCollection

def plot_ecdf_pit(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    method: Literal["pot_c", "prit_c", "piet_c", "envelope"] = ...,
    envelope_prob: float | None = ...,
    coverage: bool = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "suspicious_points",
            "p_value_text",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "suspicious_points",
            "p_value_text",
            "xlabel",
            "ylabel",
            "title",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["ecdf_pit"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
