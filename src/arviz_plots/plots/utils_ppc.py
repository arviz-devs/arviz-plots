"""Utility functions for PPC distribution and PIT plots."""

import warnings
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.validate import validate_or_use_rcparam, validate_sample_dims
from arviz_stats.base import array_stats

from arviz_plots.plots.utils import process_group_variables_coords
from arviz_plots.plots.utils_plot_types import (
    warn_if_binary,
    warn_if_discrete,
    warn_if_prior_predictive,
)


def prepare_ppc_dist_data(
    dt,
    *,
    var_names,
    filter_vars,
    group,
    coords,
    sample_dims,
    kind,
    num_samples,
    plot_collection,
    backend,
    stats,
    require_observed=False,
    warn_discrete_dist=False,
):
    """Prepare data for PPC distribution and PIT plots."""
    kind = validate_or_use_rcparam(kind, "plot.density_kind")

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    predictive_dist = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    sample_dims = validate_sample_dims(sample_dims, data=predictive_dist)
    pp_dims = [dim for dim in predictive_dist.dims if dim not in sample_dims]

    try:
        observed_dist = process_group_variables_coords(
            dt,
            group="observed_data",
            var_names=var_names,
            filter_vars=filter_vars,
            coords=coords,
        )
    except KeyError as err:
        if require_observed:
            raise err
        observed_dist = None

    warn_if_binary(observed_dist, predictive_dist)
    if warn_discrete_dist:
        warn_if_discrete(observed_dist, predictive_dist, kind)

    warn_if_prior_predictive(group)

    n_pp_samples = int(
        np.prod([predictive_dist.sizes[dim] for dim in sample_dims if dim in predictive_dist.dims])
    )
    if num_samples > n_pp_samples:
        num_samples = n_pp_samples
        warnings.warn("num_samples is larger than the number of predictive samples.")

    if kind == "dot" and stats is not None:
        stats.setdefault("predictive_dist", {"top_only": True})

    rng = np.random.default_rng(4214)
    pp_sample_ix = rng.choice(n_pp_samples, size=num_samples, replace=False)
    predictive_dist_sub = predictive_dist.stack(sample=sample_dims).isel(sample=pp_sample_ix)

    return plot_bknd, pp_dims, sample_dims, predictive_dist, predictive_dist_sub, observed_dist


def get_suspicious_mask_ds(observed_dist, pit_dt, alpha, gamma, method):
    """Compute most suspicious observations based on PIT uniformity test results."""
    pit_ds = pit_dt["ecdf_pit"].ds[list(observed_dist.data_vars)]
    p_values, _, shapley_vals = pit_ds.azstats.uniformity_test(dim=pit_ds.dims, method=method)

    highlight = (shapley_vals > gamma) & (p_values < alpha)
    return xr.Dataset(
        {
            var: highlight[var].rename({"pit_dim": next(iter(pit_ds[var].dims))})
            for var in highlight.data_vars
        }
    )


def get_ppc_pit(predictive_dist, observed_dist, sample_dims, coverage, method):
    """Compute PIT ECDF values, with optional coverage transformation and Pareto tail refinement.

    The probability of the posterior predictive being less than or equal to the observed data
    should be uniformly distributed. This function computes the PIT values with
    Generalized Pareto Distribution tail refinement.

    Parameters
    ----------
    predictive_dist : xarray.Dataset
        The posterior predictive distribution.
    observed_dist : xarray.Dataset
        The observed data.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce.
    coverage : bool
        Whether to compute the coverage.
    method : {"envelope", "pot_c", "prit_c", "piet_c"}
        The method to use for PIT computation.
    """
    rng = np.random.default_rng(214)

    pareto_pit = method in ("pot_c", "piet_c")

    dictio = {}
    for var in observed_dist.data_vars:
        if pareto_pit:
            pred_stacked = predictive_dist[var].stack(__sample__=sample_dims)
            vals = xr.apply_ufunc(
                array_stats._pareto_pit_vec,  # pylint: disable=protected-access
                pred_stacked,
                observed_dist[var],
                input_core_dims=[["__sample__"], []],
                output_core_dims=[[]],
                vectorize=False,
                kwargs={"rng": rng},
            )
        else:
            vals_less = (predictive_dist[var] < observed_dist[var]).mean(sample_dims)
            vals_eq = (predictive_dist[var] == observed_dist[var]).mean(sample_dims)
            urvs = rng.uniform(size=vals_less.values.shape)
            vals = vals_less + urvs * vals_eq

        if coverage:
            vals = 2 * np.abs(vals - 0.5)

        dictio[var] = vals

    return xr.DataTree.from_dict({"ecdf_pit": xr.Dataset(dictio)})
