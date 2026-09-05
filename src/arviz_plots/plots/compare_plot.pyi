# File generated with docstub

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal

import numpy as np
import pandas
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.validate import validate_dict_argument
from xarray import Dataset, DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import get_visual_kwargs

from .plot_collection import PlotCollection

def plot_compare(
    cmp_df: pandas.DataFrame,
    *,
    relative_scale: bool = ...,
    rotated: bool = ...,
    hide_top_model: bool = ...,
    backend: Literal["bokeh", "matplotlib", "plotly"] | None = ...,
    visuals: Mapping[
        Literal[
            "point_estimate",
            "error_bar",
            "ref_line",
            "ref_band",
            "similar_line",
            "labels",
            "title",
            "ticklabels",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
