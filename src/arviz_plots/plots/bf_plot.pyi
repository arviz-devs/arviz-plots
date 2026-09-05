# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import arviz_base
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.validate import validate_dict_argument
from arviz_stats.bayes_factor import bayes_factor
from xarray import DataTree

from arviz_plots.plots.prior_posterior_plot import plot_prior_posterior
from arviz_plots.plots.utils import add_lines, filter_aes, get_visual_kwargs
from arviz_plots.visuals import annotate_xy

from .plot_collection import PlotCollection

def plot_bf(
    dt: DataTree,
    var_names: str,
    *,
    sample_dims: str | Sequence[Hashable] | None = ...,
    ref_val: int = ...,
    kind: Literal["kde", "hist", "dot", "ecdf"] | None = ...,
    bf_type: Literal["BF10", "BF01"] = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "ref_line",
            "ref_value_text",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "dist",
            "ref_line",
            "ref_value_text",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
