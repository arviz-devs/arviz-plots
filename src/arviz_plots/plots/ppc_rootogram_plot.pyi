# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import (
    validate_ci_prob,
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from arviz_stats.helper_stats import point_interval_unique, point_unique
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.plots.utils_plot_types import raise_if_continuous, warn_if_binary
from arviz_plots.visuals import (
    ci_line_y,
    grid,
    labelled_title,
    labelled_x,
    labelled_y,
    scatter_xy,
    set_yscale,
)

from .plot_collection import PlotCollection

def plot_ppc_rootogram(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    ci_prob: float | None = ...,
    point_estimate: Literal["mean", "median", "mode"] | None = ...,
    yscale: str = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "predictive_markers",
            "observed_markers",
            "credible_interval",
            "xlabel",
            "ylabel",
            "grid",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "predictive_markers",
            "observed_markers",
            "credible_interval",
            "xlabel",
            "ylabel",
            "grid",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
