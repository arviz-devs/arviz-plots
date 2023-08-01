# pylint: disable=unused-argument
"""ArviZ intermediate level visuals."""
from importlib import import_module

import xarray as xr


def line_xy(da, target, backend, **kwargs):
    """Plot a line x vs y.

    The input argument `da` is split into x and y using the dimension ``plot_axis``.
    """
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(da.sel(plot_axis="x"), da.sel(plot_axis="y"), target, **kwargs)


def line_x(da, target, backend, y=None, **kwargs):
    """Plot a line along the x axis (y constant)."""
    if y is None:
        y = xr.zeros_like(da)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(da, y, target, **kwargs)


def scatter_x(da, target, backend, y=None, **kwargs):
    """Plot a dot/rug/scatter along the x axis (y constant)."""
    if y is None:
        y = xr.zeros_like(da)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.scatter(da, y, target, **kwargs)


def point_estimate_text(
    da, target, backend, *, point_estimate, x=None, y=None, point_label="x", **kwargs
):
    """Annotate a point estimate."""
    if y is None and x is None:
        x = da.sel(plot_axis="x")
        y = da.sel(plot_axis="y")
        point = x if point_label == "x" else y
        text = f"{point.item():.3g} {point_estimate}"
    elif x is None:
        x = da if "plot_axis" not in da.dims else da.sel(plot_axis="x")
        text = f"{x.item():.3g} {point_estimate}"
    elif y is None:
        y = da if "plot_axis" not in da.dims else da.sel(plot_axis="y")
        text = f"{y.item():.3g} {point_estimate}"
    else:
        raise ValueError("Found both x and y arguments, only one of them or neither are allowed")
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.text(
        x,
        y,
        text,
        target,
        **kwargs,
    )


def labelled_title(da, target, backend, *, labeller, var_name, sel, isel, **kwargs):
    """Add a title label to a plot using an ArviZ labeller."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.title(labeller.make_label_vert(var_name, sel, isel), target, **kwargs)


def remove_axis(da, target, backend, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    plot_backend.remove_axis(target, **kwargs)
