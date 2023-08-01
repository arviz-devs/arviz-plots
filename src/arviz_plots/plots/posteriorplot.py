"""Posterior plot code."""
from importlib import import_module

import arviz_stats  # pylint: disable=unused-import
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotMuseum


## probably move to utils.py or similar ##
def filter_aes(pc, aes_map, artist, sample_dims):
    artist_aes = aes_map.get(artist, {})
    pc_aes = pc._aes.keys()
    ignore_aes = set(pc_aes).difference(artist_aes)
    _, all_loop_dims = pc.update_aes(ignore_aes=ignore_aes)
    artist_dims = [dim for dim in sample_dims if dim not in all_loop_dims]
    return artist_dims, artist_aes, ignore_aes


### to go in visuals module ###
def line_xy(da, target, backend, **kwargs):
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(da.sel(plot_axis="x"), da.sel(plot_axis="y"), target, **kwargs)


def line_x(da, target, backend, y=None, **kwargs):
    if y is None:
        y = xr.zeros_like(da)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(da, y, target, **kwargs)


def scatter_x(da, target, backend, y=None, **kwargs):
    if y is None:
        y = xr.zeros_like(da)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.scatter(da, y, target, **kwargs)


def point_estimate_text(da, target, backend, *, point_estimate, y=None, **kwargs):
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.text(
        da.sel(plot_axis="x"),
        da.sel(plot_axis="y"),
        f"{da.sel(plot_axis='x').item():.3g} {point_estimate}",
        target,
        **kwargs,
    )


def labelled_title(da, target, backend, *, labeller, var_name, sel, isel, **kwargs):
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.title(labeller.make_label_vert(var_name, sel, isel), target, **kwargs)


def remove_axis(da, target, backend, **kwargs):
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    plot_backend.remove_axis(target, **kwargs)


### end of visuals ###


def plot_posterior(
    ds,
    coords=None,
    labeller=None,
    sample_dims=None,
    kind=None,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    plot_museum=None,
    backend=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot 1D marginal densities in the style of John K. Kruschke’s book."""
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
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if stats_kwargs is None:
        stats_kwargs = {}

    if plot_museum is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in ds.dims if dim not in sample_dims]
        )
        plot_museum = PlotMuseum.wrap(
            ds,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {"kde": plot_museum._aes.keys()}
    if labeller is None:
        labeller = BaseLabeller()

    # density
    density_dims, _, density_ignore = filter_aes(plot_museum, aes_map, "kde", sample_dims)
    if kind == "kde":
        density = ds.azstats.kde(dims=density_dims, **stats_kwargs.get("density", {}))
        plot_museum.map(
            line_xy, "kde", data=density, ignore_aes=density_ignore, **plot_kwargs.get("kde", {})
        )
    else:
        raise NotImplementedError("coming soon")

    # credible interval
    ci_dims, ci_aes, ci_ignore = filter_aes(plot_museum, aes_map, "ci", sample_dims)
    if ci_kind == "eti":
        ci = ds.azstats.eti(prob=ci_prob, dims=ci_dims, **stats_kwargs.get("ci", {}))
    elif ci_kind == "hdi":
        ci = ds.azstats.hdi(prob=ci_prob, dims=ci_dims, **stats_kwargs.get("ci", {}))
    else:
        raise NotImplementedError("coming soon")
    ci_kwargs = plot_kwargs.get("ci", {}).copy()
    if "color" not in ci_aes:
        ci_kwargs.setdefault("color", "gray")
    plot_museum.map(line_x, "ci", data=ci, ignore_aes=ci_ignore, **ci_kwargs)

    # point estimate
    pe_dims, pe_aes, pe_ignore = filter_aes(plot_museum, aes_map, "point_estimate", sample_dims)
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
    plot_museum.map(
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
    plot_museum.map(
        point_estimate_text,
        "point_estimate_text",
        data=point,
        point_estimate=point_estimate,
        ignore_aes=pe_ignore,
        **pet_kwargs,
    )

    # aesthetics
    _, title_aes, title_ignore = filter_aes(plot_museum, aes_map, "title", sample_dims)
    title_kwargs = plot_kwargs.get("title", {}).copy()
    if "color" not in title_aes:
        title_kwargs.setdefault("color", "black")
    plot_museum.map(
        labelled_title,
        "title",
        ignore_aes=title_ignore,
        subset_info=True,
        labeller=labeller,
        **title_kwargs,
    )
    plot_museum.map(remove_axis, store_artist=False, axis="y", ignore_aes=plot_museum._aes.keys())

    return plot_museum
