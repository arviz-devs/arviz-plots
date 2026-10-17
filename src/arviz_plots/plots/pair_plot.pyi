# File generated with docstub

from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams, xarray_sel_iter
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument, validate_sample_dims
from arviz_stats import kde2d
from xarray import DataTree

from arviz_plots.plot_matrix import PlotMatrix
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
    set_grid_layout,
)
from arviz_plots.visuals import (
    contour,
    contourf,
    label_plot,
    labelled_x,
    labelled_y,
    remove_matrix_axis,
    scatter_couple,
    set_ticklabel_visibility,
)

def plot_pair(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: Mapping | None = ...,
    sample_dims: Iterable | None = ...,
    levels: int | list[float] | None = ...,
    marginal: bool = ...,
    marginal_kind: Literal["kde", "hist", "ecdf", "dot"] | None = ...,
    triangle: Literal["both", "upper", "lower"] = ...,
    plot_matrix: PlotMatrix | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly", "none"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "scatter",
            "contour",
            "contourf",
            "divergence",
            "dist",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "label",
            "xlabel",
            "ylabel",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "scatter",
            "contour",
            "contourf",
            "divergence",
            "dist",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "label",
            "xlabel",
            "ylabel",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[
        Literal[
            "dist",
            "credible_interval",
            "point_estimate",
        ],
        Mapping[str, Any] | xr.Dataset,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotMatrix: ...
def _kde_couple(
    da_x: Incomplete,
    da_y: Incomplete,
    target: Incomplete,
    filled: Incomplete = ...,
    levels: Incomplete = ...,
    sample_dims: Incomplete = ...,
    **kw: Incomplete,
) -> None: ...
