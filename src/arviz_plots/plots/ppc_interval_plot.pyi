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
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    ci_bound_y,
    labelled_title,
    labelled_x,
    labelled_y,
    point_y,
)

from .plot_collection import PlotCollection

def plot_ppc_interval(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    point_estimate: Literal["mean", "median", "mode"] | None = ...,
    ci_kind: Literal["hdi", "eti"] | None = ...,
    ci_probs: tuple[float, float] | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly", "none"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "trunk",
            "twig",
            "observed_markers",
            "prediction_markers",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str] | bool,
    ] = ...,
    visuals: Mapping[
        Literal[
            "trunk",
            "twig",
            "observed_markers",
            "prediction_markers",
            "xlabel",
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[
        Literal["trunk", "twig", "point_estimate"], Mapping[str, Any] | xr.Dataset
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
def _plot_interval(
    ds_predictive: Incomplete,
    ci_trunk: Incomplete,
    ci_twig: Incomplete,
    point: Incomplete,
    observed_data: Incomplete,
    sample_dims: Incomplete,
    group: Incomplete,
    plot_collection: Incomplete,
    backend: Incomplete,
    pc_kwargs: Incomplete,
    plot_bknd: Incomplete,
    visuals: Incomplete,
    aes_by_visuals: Incomplete,
    labeller: Incomplete,
) -> None: ...
