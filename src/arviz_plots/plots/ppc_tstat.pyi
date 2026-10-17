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
from arviz_base.validate import validate_dict_argument, validate_sample_dims
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.plots.utils_plot_types import warn_if_binary
from arviz_plots.visuals import scatter_x

from .plot_collection import PlotCollection

def plot_ppc_tstat(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    t_stat: str = ...,
    kind: Literal["kde", "hist", "ecdf", "dot"] | None = ...,
    point_estimate: Literal["mean", "median", "mode"] | None = ...,
    ci_kind: Literal["eti", "hdi"] | None = ...,
    ci_prob: float | None = ...,
    data_pairs: dict | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "observed_tstat",
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
            "observed_tstat",
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
