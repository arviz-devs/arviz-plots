# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import arviz_base
import xarray as xr
from _typeshed import Incomplete
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from xarray import DataTree

from arviz_plots.plots.ecdf_plot import plot_ecdf_pit
from arviz_plots.plots.utils import process_group_variables_coords
from arviz_plots.plots.utils_plot_types import warn_if_binary, warn_if_prior_predictive
from arviz_plots.plots.utils_ppc import get_ppc_pit

from .plot_collection import PlotCollection

def plot_ppc_pit(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    method: Literal["pot_c", "prit_c", "piet_c", "envelope"] = ...,
    envelope_prob: float | None = ...,
    coverage: bool = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "suspicious_points",
            "p_value_text",
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
            "suspicious_points",
            "p_value_text",
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
