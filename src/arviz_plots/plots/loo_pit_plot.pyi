# File generated with docstub

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import arviz_base
import xarray as xr
from _typeshed import Incomplete
from arviz_base import convert_to_datatree
from arviz_base.validate import validate_dict_argument
from arviz_stats.loo import loo_pit
from xarray import DataTree

from arviz_plots.plots.ecdf_plot import plot_ecdf_pit

from .plot_collection import PlotCollection

def plot_loo_pit(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: Incomplete = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    method: Literal["pot_c", "prit_c", "piet_c"] = ...,
    envelope_prob: float | None = ...,
    coverage: bool = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["ecdf_pit"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
