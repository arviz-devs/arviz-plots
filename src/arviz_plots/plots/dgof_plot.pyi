# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import arviz_base
import arviz_stats
import xarray as xr
from _typeshed import Incomplete
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from arviz_stats.ecdf_utils import (
    compute_pit_for_histogram,
    compute_pit_for_kde,
    compute_pit_for_qds,
)
from xarray import DataTree

from arviz_plots.plots.ecdf_plot import plot_ecdf_pit
from arviz_plots.plots.utils import process_group_variables_coords

from .plot_collection import PlotCollection

def plot_dgof(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    kind: Literal["kde", "hist", "dot"] | None = ...,
    method: Literal["pot_c", "prit_c", "piet_c", "envelope"] = ...,
    envelope_prob: float | None = ...,
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
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["dist", "ecdf_pit"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
