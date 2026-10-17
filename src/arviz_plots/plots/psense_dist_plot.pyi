# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from arviz_stats.psense import power_scale_dataset
from xarray import Dataset, DataTree, concat

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    get_visual_kwargs,
    process_group_variables_coords,
    set_grid_layout,
)

from .plot_collection import PlotCollection

def plot_psense_dist(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    prior_var_names: str | None = ...,
    likelihood_var_names: str | None = ...,
    prior_coords: dict | None = ...,
    likelihood_coords: dict | None = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    alphas: tuple[float] | None = ...,
    kind: Literal["kde", "hist", "dot", "ecdf"] | None = ...,
    point_estimate: Literal["mean", "median", "mode"] | None = ...,
    ci_kind: Literal["eti", "hdi"] | None = ...,
    ci_prob: float | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
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
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "title",
            "rug",
            "legend",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[
        Literal["dist", "credible_interval", "point_estimate"],
        Mapping[str, Any] | Dataset,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
