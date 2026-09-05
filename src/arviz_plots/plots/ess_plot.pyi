# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import arviz_stats
import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument, validate_sample_dims
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
    annotate_xy,
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
    trace_rug,
)

from .plot_collection import PlotCollection

def plot_ess(
    dt: DataTree | dict[str, DataTree],
    *,
    var_names: str | Sequence[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    kind: Literal["local", "quantile"] = ...,
    relative: bool = ...,
    rug: bool = ...,
    rug_kind: str = ...,
    n_points: int = ...,
    extra_methods: bool = ...,
    min_ess: int = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "ess",
            "rug",
            "mean",
            "mean_text",
            "sd",
            "sd_text",
            "min_ess",
            "title",
            "xlabel",
            "ylabel",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "ess",
            "rug",
            "mean",
            "mean_text",
            "sd",
            "sd_text",
            "min_ess",
            "title",
            "xlabel",
            "ylabel",
            "legend",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["ess", "mean", "sd"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
