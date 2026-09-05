# File generated with docstub

from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import dataset_to_dataarray, rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument, validate_sample_dims
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
)
from arviz_plots.visuals import multiple_lines, set_xticks

from .plot_collection import PlotCollection

def plot_parallel(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: Mapping | None = ...,
    sample_dims: Iterable | None = ...,
    norm_method: Literal[None, "normal", "minmax", "rank"] | None = ...,
    label_type: Literal["flat", "vert"] = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly", "none"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "line",
            "xticks",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "line",
            "xticks",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
