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
from arviz_base.validate import validate_dict_argument, validate_sample_dims
from xarray import DataTree

from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
)
from arviz_plots.visuals import vline

from .plot_collection import PlotCollection

def plot_convergence_dist(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    diagnostics: list[str] | None = ...,
    grouped: bool = ...,
    ref_line: bool = ...,
    kind: Literal["kde", "hist", "dot", "ecdf"] = ...,
    point_estimate: Incomplete = ...,
    ci_kind: Incomplete = ...,
    ci_prob: Incomplete = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "ref_line",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "dist",
            "ref_line",
            "title",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
def _compute_diagnostics(
    dt: Incomplete,
    diagnostics: Incomplete,
    sample_dims: Incomplete,
    grouped: Incomplete,
) -> None: ...
