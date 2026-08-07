# File generated with docstub

import warnings
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import arviz_stats as azs
import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base import extract, rcParams
from arviz_base.labels import MapLabeller
from arviz_base.validate import (
    validate_ci_prob,
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_prob,
    validate_sample_dims,
)
from numpy.typing import ArrayLike
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    ci_line_y,
    fill_between_y,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
)

from .plot_collection import PlotCollection

def plot_lm(
    dt: DataTree,
    *,
    x: str | Sequence[str] | None = ...,
    y: str | Sequence[str] | None = ...,
    y_obs: str | xarray.DataArray | None = ...,
    plot_dim: str | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: Mapping | None = ...,
    sample_dims: Iterable | None = ...,
    smooth: bool = ...,
    ci_kind: Literal["hdi", "eti"] | None = ...,
    ci_prob: float | ArrayLike[float] | None = ...,
    point_estimate: Literal["mean", "median", "mode"] | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh"] | None = ...,
    xlabeller: arviz_base.labels.Labeller | None = ...,
    ylabeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "pe_line",
            "ci_band",
            "ci_bounds",
            "ci_vlines",
            "observed_scatter",
            "xlabel",
            "ylabel",
        ],
        list[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "pe_line",
            "ci_band",
            "ci_bounds",
            "ci_vlines",
            "observed_scatter",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | Literal[False],
    ] = ...,
    stats: Mapping[
        Literal["credible_interval", "pe_line", "smooth"],
        Mapping[str, Any] | xr.Dataset,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
def _sort_values_by_x(values: Incomplete) -> None: ...
def _smooth_values(
    values: Incomplete, n_points: Incomplete = ..., **smooth_kwargs: Incomplete
) -> None: ...
def combine_sort_smooth(
    x_pred: Incomplete,
    plot_dim: Incomplete,
    pe_value: Incomplete,
    ci_data: Incomplete,
    smooth: Incomplete,
    smooth_kwargs: Incomplete,
) -> None: ...
