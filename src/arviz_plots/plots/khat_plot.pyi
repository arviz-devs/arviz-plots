# File generated with docstub

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import arviz_stats
import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument
from arviz_stats.base.stats_utils import calculate_khat_bin_edges

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    annotate_bin_text,
    enable_hover_labels,
    filter_aes,
    format_coords_as_labels,
    get_visual_kwargs,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    annotate_xy,
    hline,
    labelled_title,
    labelled_x,
    labelled_y,
    scatter_xy,
    set_xlim,
    set_xticks,
)

from .plot_collection import PlotCollection

def plot_khat(
    elpd_data: arviz_stats.ELPDData,
    *,
    threshold: float | None = ...,
    hover_format: str = ...,
    legend: bool | None = ...,
    color: Any | None = ...,
    marker: Any | None = ...,
    hline_values: Sequence[float] | None = ...,
    bin_format: str = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "khat",
            "threshold_text",
            "hover",
            "title",
            "xlabel",
            "ylabel",
            "ticks",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "khat",
            "hlines",
            "bin_text",
            "threshold_text",
            "hover",
            "title",
            "xlabel",
            "ylabel",
            "legend",
            "ticks",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
