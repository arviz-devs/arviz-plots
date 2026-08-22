# File generated with docstub

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.base.stats_utils import round_num
from numpy.typing import ArrayLike

from arviz_plots.plot_collection import backend_from_object

def hist(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def step_hist(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def line_xy(
    da: Incomplete,
    target: Incomplete,
    x: Incomplete = ...,
    y: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def ci_line_y(values: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def line_x(
    da: Incomplete, target: Incomplete, y: Incomplete = ..., **kwargs: Incomplete
) -> None: ...
def line(
    da: Incomplete, target: Incomplete, xname: Incomplete = ..., **kwargs: Incomplete
) -> None: ...
def multiple_lines(
    da: xarray.DataArray,
    target: Any,
    x_dim: Hashable,
    xvalues: ArrayLike | None = ...,
    **kwargs: Incomplete,
) -> Any: ...
def trace_rug(
    da: Incomplete,
    target: Incomplete,
    mask: Incomplete,
    xname: Incomplete = ...,
    y: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def scatter_x(
    da: Incomplete, target: Incomplete, y: Incomplete = ..., **kwargs: Incomplete
) -> None: ...
def point_y(
    da: Incomplete, target: Incomplete, x: Incomplete = ..., **kwargs: Incomplete
) -> None: ...
def ci_bound_y(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def scatter_xy(
    da: Incomplete,
    target: Incomplete,
    x: Incomplete = ...,
    y: Incomplete = ...,
    mask: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def scatter_couple(
    da_x: Incomplete,
    da_y: Incomplete,
    target: Incomplete,
    mask: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def contour(
    x_coords: Incomplete,
    y_coords: Incomplete,
    density: Incomplete,
    target: Incomplete,
    *,
    levels: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def contourf(
    x_coords: Incomplete,
    y_coords: Incomplete,
    density: Incomplete,
    target: Incomplete,
    *,
    levels: Incomplete = ...,
    cmap: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def ecdf_line(values: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def vline(values: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def hline(values: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def vspan(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def hspan(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def set_xlim(
    da: Incomplete, target: Incomplete, limits: Incomplete, **kwargs: Incomplete
) -> None: ...
def set_ylim(
    da: Incomplete, target: Incomplete, limits: Incomplete, **kwargs: Incomplete
) -> None: ...
def dline(
    da: Incomplete,
    target: Incomplete,
    x: Incomplete = ...,
    y: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def fill_between_y(
    da: Incomplete,
    target: Incomplete,
    *,
    x: Incomplete = ...,
    y_bottom: Incomplete = ...,
    y: Incomplete = ...,
    y_top: Incomplete = ...,
    step: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def _process_da_x_y(
    da: Incomplete, x: Incomplete, y: Incomplete, mask: Incomplete = ...
) -> None: ...
def _ensure_scalar(*args: Incomplete) -> None: ...
def annotate_xy(
    da: Incomplete,
    target: Incomplete,
    *,
    text: Incomplete,
    x: Incomplete = ...,
    y: Incomplete = ...,
    vertical_align: Incomplete = ...,
    horizontal_align: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def point_estimate_text(
    da: Incomplete,
    target: Incomplete,
    *,
    point_estimate: Incomplete,
    x: Incomplete = ...,
    y: Incomplete = ...,
    point_label: Incomplete = ...,
    round_to: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def annotate_label(
    da: Incomplete,
    target: Incomplete,
    *,
    var_name: Incomplete,
    sel: Incomplete,
    isel: Incomplete,
    x: Incomplete = ...,
    y: Incomplete = ...,
    dim: Incomplete = ...,
    labeller: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def label_plot(
    da: Incomplete,
    target: Incomplete,
    text: Incomplete = ...,
    x: Incomplete = ...,
    y: Incomplete = ...,
    lim_low: Incomplete = ...,
    lim_high: Incomplete = ...,
    labeller: Incomplete = ...,
    var_name: Incomplete = ...,
    axis_to_remove: Incomplete = ...,
    sel: Incomplete = ...,
    isel: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def set_ticklabel_visibility(
    da: Incomplete,
    target: Incomplete,
    *,
    axis: Incomplete = ...,
    visible: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def labelled_title(
    da: Incomplete,
    target: Incomplete,
    *,
    text: Incomplete = ...,
    labeller: Incomplete = ...,
    var_name: Incomplete = ...,
    sel: Incomplete = ...,
    isel: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def labelled_y(
    da: Incomplete,
    target: Incomplete,
    *,
    text: Incomplete = ...,
    labeller: Incomplete = ...,
    var_name: Incomplete = ...,
    sel: Incomplete = ...,
    isel: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def labelled_x(
    da: Incomplete,
    target: Incomplete,
    *,
    text: Incomplete = ...,
    labeller: Incomplete = ...,
    var_name: Incomplete = ...,
    sel: Incomplete = ...,
    isel: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def ticklabel_props(
    da: Incomplete, target: Incomplete, **kwargs: Incomplete
) -> None: ...
def remove_axis(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def remove_matrix_axis(
    da_x: Incomplete, da_y: Incomplete, target: Incomplete, **kwargs: Incomplete
) -> None: ...
def remove_ticks(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def set_xticks(
    da: Incomplete,
    target: Incomplete,
    values: Incomplete,
    labels: Incomplete,
    **kwargs: Incomplete,
) -> None: ...
def set_xscale(
    da: Incomplete, target: Incomplete, scale: Incomplete, **kwargs: Incomplete
) -> None: ...
def set_yscale(
    da: Incomplete, target: Incomplete, scale: Incomplete, **kwargs: Incomplete
) -> None: ...
def grid(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
