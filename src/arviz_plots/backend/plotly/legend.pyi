# File generated with docstub

from functools import partial

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import xarray_sel_iter
from plotly.graph_objects import Bar, Scatter

from arviz_plots.backend.plotly.core import expand_aesthetic_aliases

from .plot_collection import PlotCollection

def dealiase_line_kwargs(**kwargs: Incomplete) -> None: ...
def _trace_matcher(trace: Incomplete, target_viz: Incomplete) -> None: ...

LINE_SUBKEYS: Incomplete

def legend(
    plot_collection: PlotCollection,
    kwarg_list: list,
    label_list: list,
    title: str | None = ...,
    visual_type: Incomplete = ...,
    visual_kwargs: Incomplete = ...,
    legend_dim: str | tuple[str] | None = ...,
    update_visuals: bool = ...,
    **kwargs: dict,
) -> None: ...
