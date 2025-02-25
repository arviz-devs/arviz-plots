"""Contain functions for Bayesian Factor plotting."""

from importlib import import_module
from copy import copy

import xarray as xr
from arviz_base import rcParams, extract

from arviz_stats.bayes_factor import bayes_factor
from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes
from arviz_plots.visuals import vline, annotate_xy


from arviz_plots.plots.distplot import plot_dist


def plot_bf(
    dt,
    var_name,
    ref_val=0,
    kind=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Compute and plot the Bayesian Factor."""
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()
    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if not isinstance(plot_kwargs, dict):
        plot_kwargs = {}

    ref_line_kwargs = copy(plot_kwargs.get("ref_line", {}))
    if ref_line_kwargs is False:
        raise ValueError(
            "plot_kwargs['ref_line'] can't be False, use ref_val=False to remove this element"
        )

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")


    ds_prior = extract(dt, group="prior", var_names=var_name, keep_dataset=True)
    ds_posterior = extract(dt, group="posterior", var_names=var_name, keep_dataset=True)

    distribution = xr.concat([ds_prior, ds_posterior], dim="component_group").assign_coords(
        {"component_group": ["prior", "posterior"]}
    )
    if len(sample_dims) > 1:
        # sample dims will have been stacked and renamed by `power_scale_dataset`
        sample_dims = ["sample"]

    # Compute Bayes Factor using the bayes_factor function
    bf, ref_vals = bayes_factor(dt, var_name, ref_val, return_ref_vals=True)

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("color", ["component_group"])


        figsize = pc_kwargs["plot_grid_kws"].get("figsize", None)
        figsize_units = pc_kwargs["plot_grid_kws"].get("figsize_units", "inches")
        if figsize is None:
            figsize = plot_bknd.scale_fig_size(
                figsize,
                rows=1,
                cols=1,
                figsize_units=figsize_units,
            )
            figsize_units = "dots"
        pc_kwargs["plot_grid_kws"]["figsize"] = figsize
        pc_kwargs["plot_grid_kws"]["figsize_units"] = figsize_units

        plot_collection = PlotCollection.grid(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    plot_kwargs.setdefault("credible_interval", False)
    plot_kwargs.setdefault("point_estimate", False)
    plot_kwargs.setdefault("point_estimate_text", False)

    plot_collection = plot_dist(
        distribution,
        var_names=var_name,
        group=None,
        coords=None,
        sample_dims=sample_dims,
        kind=kind,
        point_estimate=None,
        ci_kind=None,
        ci_prob=None,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
        pc_kwargs=pc_kwargs,
    )


    if ref_val:
        ref_dt = xr.Dataset({var_name: xr.DataArray([ref_val])})
        _, ref_aes, ref_ignore = filter_aes(plot_collection, aes_map, "ref_line", sample_dims)
        if "color" not in ref_aes:
            ref_line_kwargs.setdefault("color", "black")
        if "linestyle" not in ref_aes:
            default_linestyle = plot_bknd.get_default_aes("linestyle", 2, {})[1]
            ref_line_kwargs.setdefault("linestyle", default_linestyle)
        if "alpha" not in ref_aes:
            ref_line_kwargs.setdefault("alpha", 0.5)

        plot_collection.map(
            vline, "ref_line", data=ref_dt, ignore_aes=ref_ignore, **ref_line_kwargs
        )



    bf_text_kwargs = copy(plot_kwargs.get("bf_text", {}))
    if bf_text_kwargs is not False:
        _, bf_text_aes, bf_text_ignore = filter_aes(
            plot_collection, aes_map, "bf_text", sample_dims
        )


    bf_dt = xr.Dataset({var_name: xr.DataArray([0])})
    plot_collection.map(
        annotate_xy,
        "sd_text",
        text=f"BF10: {bf["BF10"]:.2f}\nBF01: {bf["BF01"]:.2f}",
        x=ref_val,
        y=max(ref_vals["prior"], ref_vals["posterior"]),
        data=bf_dt,
        ignore_aes=bf_text_ignore,
        **bf_text_kwargs,
    )

    return plot_collection
