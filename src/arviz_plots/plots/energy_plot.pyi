# File generated with docstub

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_base
import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_dataset, rcParams
from xarray import Dataset, DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import filter_aes, get_visual_kwargs, set_grid_layout
from arviz_plots.visuals import labelled_title, labelled_y, scatter_xy, vline

from .plot_collection import PlotCollection

def plot_energy(
    dt: DataTree,
    *,
    sample_dims: Sequence[str] | None = ...,
    kind: Literal["kde", "hist", "dot", "ecdf"] | None = ...,
    show_bfmi: bool = ...,
    threshold: float = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "title",
            "bfmi_points",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "dist",
            "title",
            "legend",
            "remove_axis",
            "bfmi_points",
            "ref_line",
            "title",
            "ylabel",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
def _get_energy_ds(dt: Incomplete, sample_dims: Incomplete) -> tuple[Dataset, Dataset]: ...
