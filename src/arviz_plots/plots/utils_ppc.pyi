# File generated with docstub

import warnings
from collections.abc import Hashable, Sequence
from importlib import import_module
from typing import Literal

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.validate import validate_or_use_rcparam, validate_sample_dims
from arviz_stats.base import array_stats
from xarray import Dataset

from arviz_plots.plots.utils import process_group_variables_coords
from arviz_plots.plots.utils_plot_types import (
    warn_if_binary,
    warn_if_discrete,
    warn_if_prior_predictive,
)

def prepare_ppc_dist_data(
    dt: Incomplete,
    *,
    var_names: Incomplete,
    filter_vars: Incomplete,
    group: Incomplete,
    coords: Incomplete,
    sample_dims: Incomplete,
    kind: Incomplete,
    num_samples: Incomplete,
    plot_collection: Incomplete,
    backend: Incomplete,
    stats: Incomplete,
    require_observed: Incomplete = ...,
    warn_discrete_dist: Incomplete = ...,
    warn_prior_predictive: Incomplete = ...,
) -> None: ...
def get_suspicious_mask_ds(
    observed_dist: Incomplete,
    pit_dt: Incomplete,
    alpha: Incomplete,
    gamma: Incomplete,
    method: Incomplete,
) -> None: ...
def get_ppc_pit(
    predictive_dist: Dataset,
    observed_dist: Dataset,
    sample_dims: str | Sequence[Hashable],
    coverage: bool,
    method: Literal["envelope", "pot_c", "prit_c", "piet_c"],
) -> None: ...
