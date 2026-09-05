# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
from _typeshed import Incomplete
from arviz_base import extract, rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument, validate_sample_dims
from arviz_stats.psense import power_scale_dataset
from xarray import DataTree, concat

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_grid_layout,
)
from arviz_plots.visuals import (
    hline,
    labelled_title,
    labelled_x,
    line_xy,
    scatter_xy,
    set_xticks,
)

from .plot_collection import PlotCollection

def plot_psense_quantities(
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
    quantities: list[str] | None = ...,
    mcse: bool = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "prior_markers",
            "prior_lines",
            "likelihood_markers",
            "likelihood_lines",
            "mcse",
            "ticks",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "prior_markers",
            "prior_lines",
            "likelihood_markers",
            "likelihood_lines",
            "mcse",
            "ticks",
            "title",
            "legend",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
