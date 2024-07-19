# pylint: disable=unused-argument
"""Intermediate level visuals elements.

The visuals module provides backend-agnostic functionality.
That is, the functions in this module take a set of arguments,
take care of backend-agnostic processing of those arguments
and eventually they call the requested plotting backend.
"""
import warnings
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base.labels import BaseLabeller


def hist(da, target, backend, **kwargs):
    """Plot a histogram bins(as two arrays of left and right bin edges) vs bin_height('y').

    The input argument `da` is split into l_e, r_e and y using the dimension ``plot_axis``.
    """
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.hist(
        da.sel(plot_axis="histogram"),
        da.sel(plot_axis="left_edges"),
        da.sel(plot_axis="right_edges"),
        target,
        **kwargs,
    )


def line_xy(da, target, backend, x=None, y=None, **kwargs):
    """Plot a line x vs y.

    The input argument `da` is split into x and y using the dimension ``plot_axis``.
    If additional x and y arguments are provided, x and y are added to the values
    in the `da` dataset sliced along plot_axis='x' and plot_axis='y'.
    """
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    x, y = _process_da_x_y(da, x, y)
    return plot_backend.line(x, y, target, **kwargs)


def line_x(da, target, backend, y=None, **kwargs):
    """Plot a line along the x axis (y constant)."""
    if y is None:
        y = np.zeros_like(da)
    if np.asarray(y).size == 1:
        y = np.zeros_like(da) + (y.item() if hasattr(y, "item") else y)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(da, y, target, **kwargs)


def line(da, target, backend, xname=None, **kwargs):
    """Plot a line along the y axis with x being the range of len(y)."""
    if len(da.shape) != 1:
        raise ValueError(f"Expected unidimensional data but got {da.sizes}")
    yvalues = da.values
    xvalues = np.arange(len(yvalues)) if xname is None else da[xname]
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.line(xvalues, yvalues, target, **kwargs)


def trace_rug(da, target, backend, mask=None, flatten=False, xname=None, y=None, **kwargs):
    """Create a rug plot with the subset of `da` indicated by `mask`."""
    xname = xname.item() if hasattr(xname, "item") else xname
    if xname is False:
        xvalues = da
    else:
        if xname is None:
            if len(da.shape) != 1:
                raise ValueError(f"Expected unidimensional data but got {da.sizes}")
            xvalues = np.arange(len(da))
        else:
            xvalues = da[xname]
        if y is None:
            y = da.min().item()
    if flatten is not False:
        xvalues = xvalues.values.flatten()  # flatten xvalues
    if len(xvalues.shape) != 1:
        raise ValueError(f"Expected unidimensional data but got {xvalues.sizes}")
    if mask is not None:
        xvalues = xvalues[mask]
    return scatter_x(xvalues, target=target, backend=backend, y=y, **kwargs)


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


def fill_between_y(da, target, backend, *, x=None, y_bottom=None, y=None, y_top=None, **kwargs):
    """Fill the region between to given y values."""
    if "kwarg" in da.dims:
        if "x" in da.kwarg:
            x = da.sel(kwarg="x") if x is None else da.sel(kwarg="x") + x
        if "y_bottom" in da.kwarg:
            y_bottom = (
                da.sel(kwarg="y_bottom")
                if y_bottom is None
                else da.sel(kwarg="y_bottom") + y_bottom
            )
        if "y_top" in da.kwarg:
            y_top = da.sel(kwarg="y_top") if y_top is None else da.sel(kwarg="y_top") + y_top
    if y is not None:
        y_top += y
        y_bottom += y
    if np.ndim(np.squeeze(y_top)) == 0:
        y_top = np.full_like(x, y_top)
    if np.ndim(np.squeeze(y_bottom)) == 0:
        y_bottom = np.full_like(x, y_bottom)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.fill_between_y(x, y_bottom, y_top, target, **kwargs)


def _process_da_x_y(da, x, y):
    """Process da, x and y arguments into x and y values."""
    da_has_x = "plot_axis" in da.dims and "x" in da.plot_axis
    da_has_y = "plot_axis" in da.dims and "y" in da.plot_axis
    if da_has_x:
        x = da.sel(plot_axis="x") if x is None else da.sel(plot_axis="x") + x
    if da_has_y:
        y = da.sel(plot_axis="y") if y is None else da.sel(plot_axis="y") + y
    if x is None and y is None:
        raise ValueError("Unable to find values for x and y.")
    if x is None:
        x = da
    elif y is None:
        y = da
    return np.broadcast_arrays(x, y)


def _ensure_scalar(*args):
    return tuple(arg.item() if hasattr(arg, "item") else arg for arg in args)


def point_estimate_text(
    da, target, backend, *, point_estimate, x=None, y=None, point_label="x", **kwargs
):
    """Annotate a point estimate."""
    x, y = _ensure_scalar(*_process_da_x_y(da, x, y))
    point = x if point_label == "x" else y
    if np.size(point) != 1:
        raise ValueError(
            "Found non-scalar point estimate. Check aes mapping and sample_dims. "
            f"The dimensions still left to reduce/facet are {point.dims}."
        )
    text = f"{point:.3g} {point_estimate}"
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
    x, y = _ensure_scalar(*_process_da_x_y(da, x, y))
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


def labelled_y(
    da, target, backend, *, text=None, labeller=None, var_name=None, sel=None, isel=None, **kwargs
):
    """Add a y label to a plot using an ArviZ labeller."""
    if text is None and labeller is None:
        raise ValueError("Either text or labeller must be provided")
    if text is not None and labeller is not None:
        raise ValueError("Only text or labeller can be provided")
    if labeller is not None:
        text = labeller.make_label_vert(var_name, sel, isel)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.ylabel(text, target, **kwargs)


def labelled_x(
    da, target, backend, *, text=None, labeller=None, var_name=None, sel=None, isel=None, **kwargs
):
    """Add a x label to a plot using an ArviZ labeller."""
    if text is None and labeller is None:
        raise ValueError("Either text or labeller must be provided")
    if text is not None and labeller is not None:
        raise ValueError("Only text or labeller can be provided")
    if labeller is not None:
        text = labeller.make_label_vert(var_name, sel, isel)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.xlabel(text, target, **kwargs)


def ticklabel_props(da, target, backend, **kwargs):
    """Set the size of ticks."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    return plot_backend.ticklabel_props(target, **kwargs)


def remove_axis(da, target, backend, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    plot_backend.remove_axis(target, **kwargs)


def remove_ticks(da, target, backend, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    plot_backend.remove_ticks(target, **kwargs)
