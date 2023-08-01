"""Posterior plot code."""
import arviz_stats  # pylint: disable=unused-import
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes
from arviz_plots.visuals import (
    labelled_title,
    line_x,
    line_xy,
    point_estimate_text,
    remove_axis,
    scatter_x,
)


def plot_posterior(
    ds,
    sample_dims=None,
    kind=None,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot 1D marginal densities in the style of John K. Kruschke’s book.

    Parameters
    ----------
    ds : Dataset
        Input data
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot as a point
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use.
    ci_prob : float, optional
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Defaults to only mapping properties to the density representation.
    plot_kwargs : mapping, optional
        Valid keys are:

        * kde -> passed to visuals.line_xy
        * credible_interval -> passed to visuals.line_x
        * point_estimate -> passed to visuals.scatter_x
        * point_estimate_text -> passed to visuals.point_estimate_text
        * title -> passed to visuals.labelled_title

    stats_kwargs : mapping
        Valid keys are:

        * density -> passed to kde
        * credible_interval -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection
    """
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
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if stats_kwargs is None:
        stats_kwargs = {}

    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in ds.dims if dim not in sample_dims]
        )
        plot_collection = PlotCollection.wrap(
            ds,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {"kde": plot_collection.aes_set}
    if labeller is None:
        labeller = BaseLabeller()

    # density
    density_dims, _, density_ignore = filter_aes(plot_collection, aes_map, "kde", sample_dims)
    if kind == "kde":
        density = ds.azstats.kde(dims=density_dims, **stats_kwargs.get("density", {}))
        plot_collection.map(
            line_xy, "kde", data=density, ignore_aes=density_ignore, **plot_kwargs.get("kde", {})
        )
    else:
        raise NotImplementedError("coming soon")

    # credible interval
    ci_dims, ci_aes, ci_ignore = filter_aes(
        plot_collection, aes_map, "credible_interval", sample_dims
    )
    if ci_kind == "eti":
        ci = ds.azstats.eti(prob=ci_prob, dims=ci_dims, **stats_kwargs.get("credible_interval", {}))
    elif ci_kind == "hdi":
        ci = ds.azstats.hdi(prob=ci_prob, dims=ci_dims, **stats_kwargs.get("credible_interval", {}))
    else:
        raise NotImplementedError("coming soon")
    ci_kwargs = plot_kwargs.get("credible_interval", {}).copy()
    if "color" not in ci_aes:
        ci_kwargs.setdefault("color", "gray")
    plot_collection.map(line_x, "credible_interval", data=ci, ignore_aes=ci_ignore, **ci_kwargs)

    # point estimate
    pe_dims, pe_aes, pe_ignore = filter_aes(plot_collection, aes_map, "point_estimate", sample_dims)
    if point_estimate == "median":
        point = ds.median(dim=pe_dims, **stats_kwargs.get("point_estimate", {}))
    elif point_estimate == "mean":
        point = ds.mean(dim=pe_dims, **stats_kwargs.get("point_estimate", {}))
    else:
        raise NotImplementedError("coming soon")
    point_density_diff = [dim for dim in density.sel(plot_axis="y").dims if dim not in point.dims]
    point_y = 0.03 * density.sel(plot_axis="y", drop=True).max(dim=["kde_dim"] + point_density_diff)
    point = xr.concat((point, point_y), dim="plot_axis").assign_coords(plot_axis=["x", "y"])

    pe_kwargs = plot_kwargs.get("point_estimate", {}).copy()
    if "color" not in pe_aes:
        pe_kwargs.setdefault("color", "darkcyan")
    plot_collection.map(
        scatter_x,
        "point_estimate",
        data=point.sel(plot_axis="x"),
        ignore_aes=pe_ignore,
        **pe_kwargs,
    )
    pet_kwargs = plot_kwargs.get("point_estimate_text", {}).copy()
    if "color" not in pe_aes:
        pet_kwargs.setdefault("color", "darkcyan")
    pet_kwargs.setdefault("horizontal_align", "center")
    pet_kwargs.setdefault("point_label", "x")
    plot_collection.map(
        point_estimate_text,
        "point_estimate_text",
        data=point,
        point_estimate=point_estimate,
        ignore_aes=pe_ignore,
        **pet_kwargs,
    )

    # aesthetics
    _, title_aes, title_ignore = filter_aes(plot_collection, aes_map, "title", sample_dims)
    title_kwargs = plot_kwargs.get("title", {}).copy()
    if "color" not in title_aes:
        title_kwargs.setdefault("color", "black")
    plot_collection.map(
        labelled_title,
        "title",
        ignore_aes=title_ignore,
        subset_info=True,
        labeller=labeller,
        **title_kwargs,
    )
    plot_collection.map(
        remove_axis, store_artist=False, axis="y", ignore_aes=plot_collection.aes_set
    )

    return plot_collection
