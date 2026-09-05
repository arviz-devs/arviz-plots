# File generated with docstub

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import arviz_base
import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base.validate import validate_dict_argument, validate_sample_dims
from arviz_stats.loo import loo
from scipy.special import logsumexp
from xarray import DataTree

from arviz_plots.plots.pava_calibration_plot import plot_ppc_pava
from arviz_plots.plots.utils import _var_names

from .plot_collection import PlotCollection

def plot_loo_pava(
    dt: DataTree,
    *,
    var_names: str | list[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    data_type: str = ...,
    ci_prob: float | None = ...,
    plot_collection: PlotCollection | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    labeller: arviz_base.labels.Labeller | None = ...,
    aes_by_visuals: Mapping[
        Literal[
            "lines",
            "markers",
            "reference_line",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = ...,
    visuals: Mapping[
        Literal[
            "lines",
            "markers",
            "reference_line",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
def _loo_resample(
    dt: Incomplete, sample_dims: Incomplete, resolved_var_names: Incomplete
) -> None: ...
