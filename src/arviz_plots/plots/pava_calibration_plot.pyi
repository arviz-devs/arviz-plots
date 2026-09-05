# File generated with docstub

import warnings
from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_ci_prob, validate_dict_argument
from arviz_stats.helper_stats import isotonic_fit
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, get_visual_kwargs, set_wrap_layout
from arviz_plots.plots.utils_plot_types import warn_if_prior_predictive
from arviz_plots.visuals import (
    dline,
    fill_between_y,
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
)

from .plot_collection import PlotCollection

def plot_ppc_pava(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    data_type: str = ...,
    ci_prob: float | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "lines",
            "markers",
            "reference_line",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "lines",
            "markers",
            "reference_line",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
