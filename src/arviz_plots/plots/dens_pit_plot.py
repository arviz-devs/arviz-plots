"""pit density diagnostic."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr

from arviz_plots.plots.ecdf_plot import plot_ecdf_pit


def plot_dens_pit(
    dt,
    ci_prob=0.99,
    kind="kde",
    coverage=False,
    var_names=None,
    filter_vars=None,  # pylint: disable=unused-argument
    group="posterior",
    coords=None,
    sample_dims="sample",
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[Literal["dist", "ecdf_pit"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    r"""PIT Δ-ECDF density diagnostic with simultaneous confidence envelope.

    Evaluates how well a density-based visualization (e.g., KDE, histogram, or quantile dot plot)
    represents the underlying data.
    It applies the Probability Integral Transform (PIT), computing for each observation the
    cumulative probability under the estimated density. If the estimated density matches the
    true data-generating process, the PIT values should be uniformly distributed on [0, 1].


    .. [1] Säilynoja et al. *Recommendations for visual predictive checks in Bayesian workflow*.
        (2025) arXiv preprint https://arxiv.org/abs/2503.01509

    """
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if stats is None:
        stats = {}
    else:
        stats = stats.copy()

    if var_names is None:
        var_names = list(dt[group].data_vars)

    dt_group = dt[group]

    if kind == "hist":
        hist_dt = dt_group.azstats.histogram(**stats.get("dist", {}))
        new_dt = _compute_pit_for_histogram(dt_group, hist_dt, var_names=var_names)
    elif kind == "kde":
        kde_dt = dt_group.azstats.kde(**stats.get("dist", {}))
        new_dt = _compute_pit_for_kde(dt_group, kde_dt, var_names=var_names)
    elif kind == "qds":
        qd_dt = dt_group.azstats.qds(**stats.get("dist", {}))
        new_dt = _compute_pit_for_qds(dt_group, qd_dt, var_names=var_names)
    else:
        raise ValueError(f"Kind {kind} not supported. Choose from 'hist', 'kde', or 'qds'.")

    visuals.setdefault("ylabel", {})
    visuals.setdefault("xlabel", {"text": "PIT"})

    plot_collection = plot_ecdf_pit(
        new_dt,
        coords=coords,
        sample_dims="sample",
        ci_prob=ci_prob,
        coverage=coverage,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats.get("ecdf_pit", {}),
        **pc_kwargs,
    )

    return plot_collection


def _compute_pit_for_histogram(dt_group, hist_dt, var_names):
    def _pit_for_hist(values, left_edges, right_edges, histogram):
        bin_edges = np.append(left_edges, right_edges[-1])
        dx = np.diff(bin_edges)

        xmin = bin_edges[:-1]
        xmax = bin_edges[1:]
        cdf = np.concatenate([[0.0], np.cumsum(histogram * dx)])

        pit = np.zeros_like(values, dtype=float)
        pit[values >= xmax[-1]] = 1.0

        inside = (values > xmin[0]) & (values < xmax[-1])
        xi = values[inside]
        idx = np.searchsorted(xmax, xi, side="right") - 1
        pit[inside] = cdf[idx] + (xi - xmin[idx]) * histogram[idx]

        return pit

    pit_results = {}
    for var_name in var_names:
        hist_data = hist_dt[var_name]
        left_edges = hist_data.sel(plot_axis="left_edges")
        right_edges = hist_data.sel(plot_axis="right_edges")
        histogram = hist_data.sel(plot_axis="histogram")

        sample = dt_group[var_name].stack(sample=("chain", "draw"))

        pit_da = xr.apply_ufunc(
            _pit_for_hist,
            sample,
            left_edges,
            right_edges,
            histogram,
            input_core_dims=[
                [],
                ["hist_dim_" + var_name],
                ["hist_dim_" + var_name],
                ["hist_dim_" + var_name],
            ],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        pit_results[var_name] = pit_da

    return xr.Dataset(pit_results)


def _compute_pit_for_kde(dt_group, kde_dt, var_names):
    pit_results = {}
    for var_name in var_names:
        kde_data = kde_dt[var_name]
        grid = kde_data.sel(plot_axis="x")
        density = kde_data.sel(plot_axis="y")

        dx = grid.diff("kde_dim").isel(kde_dim=0)
        cdf = (density.cumsum("kde_dim") * dx).clip(0.0, 1.0)

        sample = dt_group[var_name].stack(sample=("chain", "draw"))

        def interp_cdf(sample, grid, cdf):
            return np.interp(sample, grid, cdf, left=0.0, right=1.0)

        pit = xr.apply_ufunc(
            interp_cdf,
            sample,
            grid,
            cdf,
            input_core_dims=[[], ["kde_dim"], ["kde_dim"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        pit_results[var_name] = pit

    return xr.Dataset(pit_results)


def _compute_pit_for_qds(dt_group, qds_dt, var_names):
    rng = np.random.default_rng(42)

    def _pit_f_for_qds(values, quantile_positions, radius, nqds):
        pit = np.zeros_like(values, dtype=float)

        for i, xi in enumerate(values):
            left_edges = quantile_positions - radius
            right_edges = quantile_positions + radius

            mask = left_edges > xi
            if np.any(mask):
                q_max = np.argmax(mask)
            else:
                q_max = nqds

            if q_max > 0:
                target_position = quantile_positions[q_max - 1]
                same_position = quantile_positions == target_position
                candidates_mask = same_position & (right_edges >= xi)
                if np.any(candidates_mask):
                    q_min = np.argmax(candidates_mask)
                else:
                    q_min = q_max - 1 if q_max > 0 else 0
            else:
                q_min = 0

            pit[i] = rng.uniform(q_min, q_max) / nqds

        return pit

    pit_results = {}
    for var_name in var_names:
        qds_data = qds_dt[var_name]

        quantile_positions = qds_data.sel(plot_axis="x").values

        radius = qds_data.coords[f"radius_{var_name}"].values
        nqds = qds_data.sizes["qd_dim"]

        sample = dt_group[var_name].stack(sample=("chain", "draw"))

        pit_da = xr.apply_ufunc(
            _pit_f_for_qds,
            sample,
            quantile_positions,
            radius,
            nqds,
            input_core_dims=[["sample"], ["qd_dim"], [], []],
            output_core_dims=[["sample"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        pit_results[var_name] = pit_da

    return xr.Dataset(pit_results)
