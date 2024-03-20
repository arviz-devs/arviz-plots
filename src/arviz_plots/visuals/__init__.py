# pylint: disable=unused-argument
"""Intermediate level visuals elements.

The visuals module provides backend-agnostic functionality.
That is, the functions in this module take a set of arguments,
take care of backend-agnostic processing of those arguments
and eventually they call the requested plotting backend.
"""
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base.labels import BaseLabeller


def line_xy(da, target, backend, **kwargs):
    """Plot a line x vs y.

    The input argument `da` is split into x and y using the dimension ``plot_axis``.
    """
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(da.sel(plot_axis="x"), da.sel(plot_axis="y"), target, **kwargs)


def line_x(da, target, backend, y=None, **kwargs):
    """Plot a line along the x axis (y constant)."""
    if y is None:
        y = np.zeros_like(da)
    if np.asarray(y).size == 1:
        y = np.zeros_like(da) + (y.item() if hasattr(y, "item") else y)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(da, y, target, **kwargs)


def line(da, target, backend, coord=None, **kwargs):
    """Plot a line along the y axis with x being the range of len(y)."""
    if len(da.shape) != 1:
        raise ValueError(f"Expected unidimensional data but got {da.sizes}")
    yvalues = da.values
    xvalues = np.arange(len(yvalues)) if coord is None else da.coords[coord]
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(xvalues, yvalues, target, **kwargs)


def trace_rug(da, target, backend, mask, coord=None, y=None, **kwargs):
    """Create a rug plot with the subset of `da` indicated by `mask`."""
    if len(da.shape) != 1:
        raise ValueError(f"Expected unidimensional data but got {da.sizes}")
    xvalues = np.arange(len(da)) if coord is None else da.coords[coord]
    if y is None:
        y = da.min().item()
    return scatter_x(xvalues[mask], target=target, backend=backend, y=y, **kwargs)


def scatter_x(da, target, backend, y=None, **kwargs):
    """Plot a dot/rug/scatter along the x axis (y constant)."""
    if y is None:
        y = np.zeros_like(da)
    if np.asarray(y).size == 1:
        y = np.zeros_like(da) + (y.item() if hasattr(y, "item") else y)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.scatter(da, y, target, **kwargs)


def ecdf_line(values, target, backend, **kwargs):
    """Plot an ecdf line."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(values.sel(plot_axis="x"), values.sel(plot_axis="y"), target, **kwargs)


def point_estimate_text(
    da, target, backend, *, point_estimate, x=None, y=None, point_label="x", **kwargs
):
    """Annotate a point estimate."""
    da_has_x = "plot_axis" in da.dims and "x" in da.plot_axis
    da_has_y = "plot_axis" in da.dims and "y" in da.plot_axis
    if da_has_x:
        x = da.sel(plot_axis="x") if x is None else da.sel(plot_axis="x") + x
    if da_has_y:
        y = da.sel(plot_axis="y") if y is None else da.sel(plot_axis="y") + y
    point = x if point_label == "x" else y
    if point.size != 1:
        raise ValueError(
            "Found non-scalar point estimate. Check aes mapping and sample_dims. "
            f"The dimensions still left to reduce/facet are {point.dims}."
        )
    text = f"{point.item():.3g} {point_estimate}"
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.text(
        x,
        y,
        text,
        target,
        **kwargs,
    )


def annotate_label(
    da, target, backend, *, var_name, sel, isel, x=None, y=None, dim=None, labeller=None, **kwargs
):
    """Annotate a dimension or aesthetic property."""
    da_has_x = "plot_axis" in da.dims and "x" in da.plot_axis
    da_has_y = "plot_axis" in da.dims and "y" in da.plot_axis
    if da_has_x:
        x = da.sel(plot_axis="x") if x is None else da.sel(plot_axis="x") + x
    if da_has_y:
        y = da.sel(plot_axis="y") if y is None else da.sel(plot_axis="y") + y
    if x is None and y is None:
        raise ValueError("Unable to find values for x and y.")
    if x is None:
        x = da.item()
    elif y is None:
        y = da.item()
    if labeller is None:
        labeller = BaseLabeller()
    if dim is None:
        text = labeller.make_label_flat(var_name, sel, isel)
    else:
        sel = {key: value for key, value in sel.items() if key == dim}
        isel = {key: value for key, value in isel.items() if key == dim}
        text = labeller.sel_to_str(sel, isel)
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


def labelled_y(da, target, backend, *, labeller, var_name, sel, isel, **kwargs):
    """Add a y label to a plot using an ArviZ labeller."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.ylabel(labeller.make_label_vert(var_name, sel, isel), target, **kwargs)


def labelled_x(da, target, backend, *, labeller, var_name, sel, isel, **kwargs):
    """Add a x label to a plot using an ArviZ labeller."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.xlabel(labeller.make_label_vert(var_name, sel, isel), target, **kwargs)


def xticks(da, target, backend, *, ticks, labels=None, **kwargs):
    """Add x ticks and labels to a plot."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.xticks(ticks, labels, target, **kwargs)


def yticks(da, target, backend, *, ticks, labels=None, **kwargs):
    """Add y ticks and labels to a plot."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.yticks(ticks, labels, target, **kwargs)


def ticks_size(da, target, backend, *, value, **kwargs):
    """Set the size of ticks."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.ticks_size(value, target)


def remove_axis(da, target, backend, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    plot_backend.remove_axis(target, **kwargs)


def remove_ticks(da, target, backend, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    plot_backend.remove_ticks(target, **kwargs)
