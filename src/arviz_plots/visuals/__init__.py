# pylint: disable=unused-argument
"""Intermediate level visuals elements.

The visuals module provides backend-agnostic functionality.
That is, the functions in this module take a set of arguments,
take care of backend-agnostic processing of those arguments
and eventually they call the requested plotting backend.
"""
import numpy as np
import xarray as xr
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import backend_from_object


def hist(da, target, **kwargs):
    """Plot a histogram bins(as two arrays of left and right bin edges) vs bin_height('y').

    The input argument `da` is split into l_e, r_e and y using the dimension ``plot_axis``.
    """
    plot_backend = backend_from_object(target)
    return plot_backend.hist(
        da.sel(plot_axis="histogram"),
        da.sel(plot_axis="left_edges"),
        da.sel(plot_axis="right_edges"),
        target,
        **kwargs,
    )


def line_xy(da, target, x=None, y=None, **kwargs):
    """Plot a line x vs y.

    The input argument `da` is split into x and y using the dimension ``plot_axis``.
    If additional x and y arguments are provided, x and y are added to the values
    in the `da` dataset sliced along plot_axis='x' and plot_axis='y'.
    """
    plot_backend = backend_from_object(target)
    x, y = _process_da_x_y(da, x, y)
    return plot_backend.line(x, y, target, **kwargs)


def ci_line_y(values, target, **kwargs):
    """Plot a line from y_bottom to y_top at given value of x."""
    plot_backend = backend_from_object(target)
    return plot_backend.ciliney(
        values.sel(plot_axis="x"),
        values.sel(plot_axis="y_bottom"),
        values.sel(plot_axis="y_top"),
        target,
        **kwargs,
    )


def line_x(da, target, y=None, **kwargs):
    """Plot a line along the x axis (y constant)."""
    if y is None:
        y = np.zeros_like(da)
    if np.asarray(y).size == 1:
        y = np.zeros_like(da) + (y.item() if hasattr(y, "item") else y)
    plot_backend = backend_from_object(target)
    return plot_backend.line(da, y, target, **kwargs)


def line(da, target, xname=None, **kwargs):
    """Plot a line along the y axis with x being the range of len(y)."""
    if len(da.shape) != 1:
        raise ValueError(f"Expected unidimensional data but got {da.sizes}")
    yvalues = da.values
    xvalues = np.arange(len(yvalues)) if xname is None else da[xname]
    plot_backend = backend_from_object(target)
    return plot_backend.line(xvalues, yvalues, target, **kwargs)


def multiple_lines(da, target, x_dim, xvalues=None, **kwargs):
    """Plot multiple lines together.

    Parameters
    ----------
    da : DataArray
        2d DataArray with `x_dim` as one of its dimensions.
    target : Any
        Object representing the target :term:`plot`
    x_dim : hashable
        Dimension of `da` to be encoded along the x axis of the plot.
    xvalues : array-like, optional
        Specific values for the positions of the data along the x axis.
        Defaults to ``da.coords[x_dim].values``
    **kwargs
        Passed to the backend function :func:`~arviz_plots.backend.none.multiple_lines`

    Returns
    -------
    Any
        Object representing the generated :term:`visual`
    """
    if da.ndim != 2:
        raise ValueError(f"DataArray must be 2D, but has dims: {da.dims}")

    if x_dim not in da.dims:
        raise ValueError(f"overlay_dim '{x_dim}' not found in DataArray dims {da.dims}")

    da = da.transpose(x_dim, ...)
    yvalues = da.values

    if xvalues is None:
        xvalues = da.coords[x_dim].values

    if len(xvalues) != yvalues.shape[0]:
        raise ValueError(
            f"xvalues length ({len(xvalues)}) does not match x-dim size ({yvalues.shape[0]})."
        )

    plot_backend = backend_from_object(target)
    return plot_backend.multiple_lines(xvalues, yvalues, target, **kwargs)


def trace_rug(da, target, mask, xname=None, y=None, **kwargs):
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
    if len(xvalues.shape) != 1:
        raise ValueError(f"Expected unidimensional data but got {xvalues.sizes}")
    return scatter_x(xvalues[mask], target=target, y=y, **kwargs)


def scatter_x(da, target, y=None, **kwargs):
    """Plot a dot/rug/scatter along the x axis (y constant)."""
    if y is None:
        y = np.zeros_like(da)
    if np.asarray(y).size == 1:
        y = np.zeros_like(da) + (y.item() if hasattr(y, "item") else y)
    plot_backend = backend_from_object(target)
    return plot_backend.scatter(da, y, target, **kwargs)


def scatter_xy(da, target, x=None, y=None, mask=None, **kwargs):
    """Plot a scatter plot x vs y.

    The input argument `da` is split into x and y using the dimension ``plot_axis``.
    If additional x and y arguments are provided, x and y are added to the values
    in the `da` dataset sliced along plot_axis='x' and plot_axis='y'.
    If a mask is provided, it is applied to both x and y values.
    """
    plot_backend = backend_from_object(target)
    x, y = _process_da_x_y(da, x, y, mask)
    return plot_backend.scatter(x, y, target, **kwargs)


def scatter_couple(da_x, da_y, target, mask=None, **kwargs):
    """Plot a scatter plot for a pairplot couple."""
    plot_backend = backend_from_object(target)
    if mask is not None:
        da_x = da_x[mask]
        da_y = da_y[mask]

    return plot_backend.scatter(da_x.values, da_y.values, target, **kwargs)


def ecdf_line(values, target, **kwargs):
    """Plot a step line."""
    plot_backend = backend_from_object(target)
    return plot_backend.step(values.sel(plot_axis="x"), values.sel(plot_axis="y"), target, **kwargs)


def vline(values, target, **kwargs):
    """Plot a vertical line that spans the whole figure independently of zoom."""
    plot_backend = backend_from_object(target)
    return plot_backend.vline(values.item(), target, **kwargs)


def hline(values, target, **kwargs):
    """Plot a horizontal line that spans the whole figure independently of zoom."""
    plot_backend = backend_from_object(target)
    return plot_backend.hline(values.item(), target, **kwargs)


def vspan(da, target, **kwargs):
    """Plot a vertical shaded region that spans the whole figure."""
    plot_backend = backend_from_object(target)
    return plot_backend.vspan(da.values[0], da.values[1], target, **kwargs)


def hspan(da, target, **kwargs):
    """Plot a vertical shaded region that spans the whole figure."""
    plot_backend = backend_from_object(target)
    return plot_backend.hspan(da.values[0], da.values[1], target, **kwargs)


def dline(da, target, x=None, y=None, **kwargs):
    """Plot a diagonal line across the x-y range."""
    plot_backend = backend_from_object(target)
    if x is None:
        x = y
    if y is None:
        y = x
    xy_min = min(np.min(x), np.min(y))
    xy_max = max(np.max(x), np.max(y))
    return plot_backend.line([xy_min, xy_max], [xy_min, xy_max], target, **kwargs)


def fill_between_y(da, target, *, x=None, y_bottom=None, y=None, y_top=None, **kwargs):
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
    plot_backend = backend_from_object(target)

    return plot_backend.fill_between_y(x, y_bottom, y_top, target, **kwargs)


def _process_da_x_y(da, x, y, mask=None):
    """Process da, x and y arguments into x and y values and apply mask if it is not None."""
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
    if mask is not None:
        x = x[mask]
        y = y[mask]
    return np.broadcast_arrays(x, y)


def _ensure_scalar(*args):
    return tuple(arg.item() if hasattr(arg, "item") else arg for arg in args)


def annotate_xy(
    da,
    target,
    *,
    text,
    x=None,
    y=None,
    vertical_align=None,
    horizontal_align=None,
    **kwargs,
):
    """Annotate a point (x, y) in a plot."""
    if vertical_align is not None:
        kwargs["vertical_align"] = (
            vertical_align.item() if hasattr(vertical_align, "item") else vertical_align
        )
    if horizontal_align is not None:
        kwargs["horizontal_align"] = (
            horizontal_align.item() if hasattr(horizontal_align, "item") else horizontal_align
        )
    x, y = _process_da_x_y(da, x, y)
    plot_backend = backend_from_object(target)
    return plot_backend.text(x, y, text, target, **kwargs)


def point_estimate_text(da, target, *, point_estimate, x=None, y=None, point_label="x", **kwargs):
    """Annotate a point estimate."""
    x, y = _ensure_scalar(*_process_da_x_y(da, x, y))
    point = x if point_label == "x" else y
    if np.size(point) != 1:
        raise ValueError(
            "Found non-scalar point estimate. Check aes mapping and sample_dims. "
            f"The dimensions still left to reduce/facet are {point.dims}."
        )
    text = f"{point:.3g} {point_estimate}"
    plot_backend = backend_from_object(target)
    return plot_backend.text(
        x,
        y,
        text,
        target,
        **kwargs,
    )


def annotate_label(
    da, target, *, var_name, sel, isel, x=None, y=None, dim=None, labeller=None, **kwargs
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
    plot_backend = backend_from_object(target)
    return plot_backend.text(
        x,
        y,
        text,
        target,
        **kwargs,
    )


def label_plot(
    da,
    target,
    text=None,
    x=0.5,
    y=0.5,
    lim_low=0,
    lim_high=1,
    labeller=None,
    var_name=None,
    axis_to_remove=False,
    sel=None,
    isel=None,
    **kwargs,
):
    """Add a label to a plot."""
    if text is None:
        if labeller is None:
            labeller = BaseLabeller()
        text = labeller.make_label_vert(var_name, sel, isel)
    x, y = _ensure_scalar(x, y)
    lim_low, lim_high = _ensure_scalar(lim_low, lim_high)
    plot_backend = backend_from_object(target)
    plot_backend.xlim((lim_low, lim_high), target)
    plot_backend.ylim((lim_low, lim_high), target)
    if axis_to_remove:
        plot_backend.remove_axis(target, axis=axis_to_remove)
    return plot_backend.text(
        x,
        y,
        text,
        target,
        **kwargs,
    )


def set_ticklabel_visibility(da, target, *, axis="both", visible=True, **kwargs):
    """Set the visibility of tick labels on a plot."""
    plot_backend = backend_from_object(target)
    return plot_backend.set_ticklabel_visibility(target, axis=axis, visible=visible, **kwargs)


def labelled_title(
    da, target, *, text=None, labeller=None, var_name=None, sel=None, isel=None, **kwargs
):
    """Add a title label to a plot using an ArviZ labeller."""
    if text is not None and labeller is not None:
        text = f"{labeller.make_label_vert(var_name, sel, isel)} ({text})"
    elif labeller is not None:
        text = labeller.make_label_vert(var_name, sel, isel)
    plot_backend = backend_from_object(target)
    return plot_backend.title(text, target, **kwargs)


def labelled_y(
    da, target, *, text=None, labeller=None, var_name=None, sel=None, isel=None, **kwargs
):
    """Add a y label to a plot using an ArviZ labeller."""
    if text is None and labeller is None:
        raise ValueError("Either text or labeller must be provided")
    if text is not None and labeller is not None:
        raise ValueError("Only text or labeller can be provided")
    if labeller is not None:
        text = labeller.make_label_vert(var_name, sel, isel)
    plot_backend = backend_from_object(target)
    return plot_backend.ylabel(text, target, **kwargs)


def labelled_x(
    da, target, *, text=None, labeller=None, var_name=None, sel=None, isel=None, **kwargs
):
    """Add a x label to a plot using an ArviZ labeller."""
    if text is None and labeller is None:
        raise ValueError("Either text or labeller must be provided")
    if text is not None and labeller is not None:
        raise ValueError("Only text or labeller can be provided")
    if labeller is not None:
        text = labeller.make_label_vert(var_name, sel, isel)
    plot_backend = backend_from_object(target)
    return plot_backend.xlabel(text, target, **kwargs)


def ticklabel_props(da, target, **kwargs):
    """Set the size of ticks."""
    plot_backend = backend_from_object(target)
    return plot_backend.ticklabel_props(target, **kwargs)


def remove_axis(da, target, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = backend_from_object(target)
    return plot_backend.remove_axis(target, **kwargs)


def remove_matrix_axis(da_x, da_y, target, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = backend_from_object(target)
    return plot_backend.remove_axis(target, **kwargs)


def remove_ticks(da, target, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = backend_from_object(target)
    return plot_backend.remove_ticks(target, **kwargs)


def set_xticks(da, target, values, labels, **kwargs):
    """Dispatch to ``set_xticks`` function in backend."""
    plot_backend = backend_from_object(target)
    return plot_backend.xticks(values, labels, target, **kwargs)


def set_y_scale(da, target, scale, **kwargs):
    """Set scale for y-axis."""
    plot_backend = backend_from_object(target)
    return plot_backend.set_y_scale(target, scale, **kwargs)


def grid(da, target, **kwargs):
    """Dispatch to ``remove_axis`` function in backend."""
    plot_backend = backend_from_object(target)
    return plot_backend.grid(target, **kwargs)
