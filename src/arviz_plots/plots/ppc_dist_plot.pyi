# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import arviz_base
import xarray as xr
from _typeshed import Incomplete
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument
from xarray import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import get_visual_kwargs, set_wrap_layout
from arviz_plots.plots.utils_ppc import prepare_ppc_dist_data

from .plot_collection import PlotCollection

def plot_ppc_dist(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    kind: Literal["auto", "kde", "hist", "ecdf", "dot"] | None = ...,
    num_samples: int = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal["predictive_dist", "observed_dist", "title"], Sequence[str]
    ] = ...,
    visuals: Mapping[
        Literal["predictive_dist", "observed_dist", "title", "remove_axis"],
        Mapping[str, Any] | bool,
    ] = ...,
    stats: Mapping[
        Literal["predictive_dist", "observed_dist"], Mapping[str, Any] | xr.Dataset
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
