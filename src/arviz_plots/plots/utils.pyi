# File generated with docstub

import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Literal

import matplotlib
import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams, references_to_dataset, xarray_sel_iter
from arviz_base.labels import BaseLabeller
from arviz_base.utils import _var_names
from arviz_stats import ecdf, histogram, kde, qds
from numpy.typing import ArrayLike, NDArray
from xarray import Dataset, DataTree

from arviz_plots.plot_collection import concat_model_dict, process_facet_dims
from arviz_plots.visuals import annotate_xy, hline, hspan, vline, vspan

from .plot_collection import PlotCollection

def get_group(
    data: DataTree, group: Hashable, allow_missing: bool = ...
) -> Dataset: ...
def get_visual_kwargs(
    visuals: Incomplete, name: Incomplete, default: Incomplete = ...
) -> dict: ...
def process_group_variables_coords(
    dt: Incomplete,
    group: Incomplete,
    var_names: Incomplete,
    filter_vars: Incomplete,
    coords: Incomplete,
    allow_dict: Incomplete = ...,
) -> None: ...
def filter_aes(
    pc: Incomplete,
    aes_by_visuals: Incomplete,
    visual: Incomplete,
    sample_dims: Incomplete,
) -> None: ...
def filter_aes_full(
    pc: Incomplete,
    aes_by_visuals: Incomplete,
    visual: Incomplete,
    sample_dims: Incomplete,
) -> tuple[list, list, Iterable, set]: ...
def set_wrap_layout(pc_kwargs: dict, plot_bknd: str, ds: Dataset) -> None: ...
def set_grid_layout(
    pc_kwargs: dict,
    plot_bknd: str,
    ds: Dataset,
    num_rows: int | None = ...,
    num_cols: int | None = ...,
) -> None: ...
def _compute_func_da(
    func: Incomplete,
    da: Incomplete,
    active_dims: Incomplete,
    reduce_dims: Incomplete,
    kwargs: Incomplete = ...,
) -> None: ...
def _compute_func(
    func: Incomplete,
    data: Incomplete,
    active_dims: Incomplete,
    reduce_dims: Incomplete,
    var_names: Incomplete = ...,
    kwargs: Incomplete = ...,
) -> None: ...
def compute_dist(
    data: Dataset,
    reduce_dims: Sequence[Hashable],
    active_dims: Sequence[Hashable],
    kind: Literal["auto", "kde", "hist", "ecdf", "dot"] | None = ...,
    stats: Mapping | None = ...,
) -> None: ...
def add_lines(
    plot_collection: PlotCollection,
    values: int,
    orientation: str = ...,
    aes_by_visuals: Mapping[str, Sequence[str]] | None = ...,
    visuals: Mapping[str, Mapping | bool] | None = ...,
    sample_dims: list | None = ...,
    ref_dim: str = ...,
    **kwargs: Mapping[str, Sequence],
) -> PlotCollection: ...
def add_bands(
    plot_collection: PlotCollection,
    values: tuple,
    orientation: str = ...,
    aes_by_visuals: Mapping[str, Sequence[str]] | None = ...,
    visuals: Mapping[str, Mapping | bool] | None = ...,
    sample_dims: list | None = ...,
    ref_dim: list | None = ...,
    **kwargs: Sequence,
) -> PlotCollection: ...
def format_coords_as_labels(
    data: xr.DataArray,
    skip_dims: str | Sequence[str] | None = ...,
    labeller: BaseLabeller | None = ...,
) -> NDArray[str]: ...
def annotate_bin_text(
    da: xr.DataArray,
    target: matplotlib.axes.Axes,
    x: float,
    y: float,
    count_da: int | xr.DataArray,
    n_da: int | xr.DataArray,
    bin_format: str,
    **kwargs: Incomplete,
) -> matplotlib.artist.Artist: ...
def enable_hover_labels(
    backend: str,
    plot_collection: PlotCollection,
    hover_format: str,
    labels: xr.DataArray,
    colors: xr.DataArray | None,
    values: xr.DataArray | None,
) -> None: ...
def hover(
    event: matplotlib.backend_bases.MouseEvent,
    annot: matplotlib.text.Annotation,
    ax: matplotlib.axes.Axes,
    scatter: matplotlib.collections.PathCollection,
    fig: matplotlib.figure.Figure,
    offsets: NDArray,
    labels: ArrayLike,
    values: ArrayLike | None,
    hover_format: str,
    colors: NDArray | None,
    offset_distance: float,
) -> None: ...
def hover_labels(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    scatter: matplotlib.collections.PathCollection,
    labels: ArrayLike,
    hover_format: str,
    colors: NDArray | None,
    values: ArrayLike | None,
) -> None: ...
def _format_hover_text(
    template: Incomplete, index: Incomplete, label: Incomplete, value: Incomplete
) -> None: ...
