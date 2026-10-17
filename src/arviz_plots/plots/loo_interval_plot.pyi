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
from arviz_stats.loo import loo_expectations
from xarray import DataTree

from arviz_plots.plots.ppc_interval_plot import _plot_interval
from arviz_plots.plots.utils import process_group_variables_coords

from .plot_collection import PlotCollection

def plot_loo_interval(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    point_estimate: Literal["mean", "median"] | None = ...,
    ci_kind: Incomplete = ...,
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
