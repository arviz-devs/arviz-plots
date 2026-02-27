# File generated with docstub

from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import numpy as np
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import labelled_title, labelled_x, line, ticklabel_props, trace_rug

from .plot_collection import PlotCollection

def plot_trace(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: Incomplete = ...,
    coords: Incomplete = ...,
    sample_dims: Iterable | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "trace",
            "divergence",
            "title",
            "xlabel",
            "ticklabels",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "trace",
            "divergence",
            "title",
            "xlabel",
            "ticklabels",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    **pc_kwargs: Mapping,
) -> PlotCollection: ...
