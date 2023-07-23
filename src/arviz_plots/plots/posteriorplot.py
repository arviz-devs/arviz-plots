"""Posterior plot code."""
from importlib import import_module

from arviz_base.labels import BaseLabeller
from arviz_base import rcParams
import arviz_stats  # pylint: disable=unused-import

from ..plot_collection import PlotMuseum

def plot_posterior(ds, coords=None, labeller=None, sample_dims=None, kind=None, point_estimate=None, ci_kind=None, ci_prob=None, plot_museum=None, backend=None):
    if coords is None:
        coords = {}
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"] if "stats.ci_kind" in rcParams else "eti"
    if point_estimate is None:
        point_estimate = rcParams["plot.point_estimate"]
    if kind is None:
        kind = rcParams["plot.density_kind"]

    if plot_museum is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        plot_museum = PlotMuseum.wrap(
            ds,
            cols=["__variable__"]+[dim for dim in ds.dims if dim not in sample_dims],
            col_wrap=6,
            backend=backend,
        )
    if labeller is None:
        labeller = BaseLabeller()

    # data processing
    if kind == "kde":
        density = ds.azstats.kde(dim=sample_dims)
    else:
        raise NotImplementedError("coming soon")
    if ci_kind == "eti":
        ci = ds.azstats.eti(prob=ci_prob, dims=sample_dims)
    elif ci_kind == "hdi":
        ci = ds.azstats.hdi(prob=ci_prob, dims=sample_dims)
    else:
        raise NotImplementedError("coming soon")

    if point_estimate == "median":
        point = ds.median(dim=sample_dims)
    elif point_estimate == "mean":
        point = ds.mean(dim=sample_dims)
    else:
        raise NotImplementedError("coming soon")


    # plotting
    plot_backend = import_module(f"arviz_plots.backend.{plot_museum.backend}")
    _, all_loop_dims = plot_museum.update_aes()
    plot_museum.allocate_artist("kde", ds, all_loop_dims)
    plot_museum.allocate_artist("credible_interval", ds, all_loop_dims)
    plot_museum.allocate_artist("point_estimate", ds, all_loop_dims)
    plot_museum.allocate_artist("point_estimate_text", ds, all_loop_dims)
    plot_museum.allocate_artist("title", ds, all_loop_dims)
    for target, var_name, sel, isel, aes_kwargs in plot_museum.plot_iterator():
        density_iter = density[var_name].sel(sel)
        ci_iter = ci[var_name].sel(sel)
        point_iter = point[var_name].sel(sel)
        plot_museum.viz[var_name]["kde"].loc[sel] = plot_backend.line(
            density_iter.sel(plot_axis="x"),
            density_iter.sel(plot_axis="y"),
            target,
            **aes_kwargs
        )
        plot_museum.viz[var_name]["credible_interval"].loc[sel] = plot_backend.line(
            ci_iter, [0, 0], target, **aes_kwargs, color="grey"
        )
        plot_museum.viz[var_name]["point_estimate"].loc[sel] = plot_backend.scatter(
            point_iter, 0, target, **aes_kwargs, color="darkcyan"
        )
        plot_museum.viz[var_name]["point_estimate_text"].loc[sel] = plot_backend.text(
            point_iter, 0.03 * density_iter.sel(plot_axis="y").max(), f"{point_iter.item():.3g} {point_estimate}", target, **aes_kwargs, color="darkcyan", horizontal_align="center"
        )
        plot_museum.viz[var_name]["title"].loc[sel] = plot_backend.title(labeller.make_label_vert(var_name, sel, isel), target, **aes_kwargs, color="black")
        plot_backend.remove_axis(target, axis="y")

    return plot_museum
