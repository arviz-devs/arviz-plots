"""Bokeh interface layer."""
import warnings

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import Title
from bokeh.plotting import figure
from bokeh.plotting import show as _show

from .legend import legend

__all__ = [
    "create_plotting_grid",
    "line",
    "scatter",
    "text",
    "title",
    "ylabel",
    "xlabel",
    "ticks_size",
    "remove_ticks",
    "remove_axis",
    "legend",
]


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()


# object creation and i/o
def show(chart):
    """Show the provided bokeh layout."""
    _show(chart)


def create_plotting_grid(
    number,
    rows=1,
    cols=1,
    squeeze=True,
    sharex=False,
    sharey=False,
    polar=False,
    subplot_kws=None,
    **kwargs,
):
    """Create a chart with a grid of plotting targets in it.

    Parameters
    ----------
    number : int
        Number of axes required
    rows, cols : int
        Number of rows and columns.
    squeeze : bool, default True
    sharex, sharey : bool, default False
    polar : bool
    subplot_kws : bool
        Passed to :func:`~bokeh.plotting.figure`
    **kwargs: dict, optional
        Passed to :func:`~bokeh.layouts.gridplot`

    Returns
    -------
    `~bokeh.layouts.gridplot` or None
    `~bokeh.plotting.figure` or ndarray of `~bokeh.plotting.figure`
    """
    if subplot_kws is None:
        subplot_kws = {}
    subplot_kws = subplot_kws.copy()

    figures = np.empty((rows, cols), dtype=object)

    if polar:
        subplot_kws.setdefault("x_axis_type", None)
        subplot_kws.setdefault("y_axis_type", None)

    for row in range(rows):
        for col in range(cols):
            if (row == 0) and (col == 0) and (sharex or sharey):
                p = figure(**subplot_kws)  # pylint: disable=invalid-name
                figures[row, col] = p
                if sharex:
                    subplot_kws["x_range"] = p.x_range
                if sharey:
                    subplot_kws["y_range"] = p.y_range
            elif row * cols + (col + 1) > number:
                figures[row, col] = None
            else:
                figures[row, col] = figure(**subplot_kws)
    if squeeze and figures.size == 1:
        return None, figures[0, 0]
    layout = gridplot(figures.tolist(), **kwargs)
    return layout, figures.squeeze() if squeeze else figures


# helper functions
def _filter_kwargs(kwargs, artist_kws):
    """Filter a dictionary to remove all keys whose values are ``unset``."""
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    return {**artist_kws, **kwargs}


# "geoms"
def line(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to bokeh for a line plot."""
    kwargs = {"color": color, "alpha": alpha, "line_width": width, "line_dash": linestyle}
    return target.line(np.atleast_1d(x), np.atleast_1d(y), **_filter_kwargs(kwargs, artist_kws))


def scatter(
    x,
    y,
    target,
    *,
    size=unset,
    marker=unset,
    alpha=unset,
    color=unset,
    facecolor=unset,
    edgecolor=unset,
    width=unset,
    **artist_kws,
):
    """Interface to bokeh for a scatter plot."""
    if color is not unset:
        if facecolor is not unset or edgecolor is not unset:
            warnings.warn(
                "color overrides facecolor and edgecolor. Their values will be ignored.",
                UserWarning,
            )
        facecolor = color
        edgecolor = color
    kwargs = {
        "size": size,
        "marker": marker,
        "line_alpha": alpha,
        "fill_alpha": alpha,
        "fill_color": facecolor,
        "line_color": edgecolor,
        "line_width": width,
    }
    return target.scatter(np.atleast_1d(x), np.atleast_1d(y), **_filter_kwargs(kwargs, artist_kws))


def text(
    x,
    y,
    string,
    target,
    *,
    size=unset,
    alpha=unset,
    color=unset,
    vertical_align=unset,
    horizontal_align=unset,
    **artist_kws,
):
    """Interface to bokeh for adding text to a plot."""
    kwargs = {
        "text_font_size": size,
        "alpha": alpha,
        "color": color,
        "text_align": horizontal_align,
        "text_baseline": vertical_align,
    }
    return target.text(
        np.atleast_1d(x),
        np.atleast_1d(y),
        np.atleast_1d(string),
        **_filter_kwargs(kwargs, artist_kws),
    )


# general plot appeareance
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to bokeh for adding a title to a plot."""
    kwargs = {"text_font_size": size, "text_color": color}
    target.title = Title(text=string, **_filter_kwargs(kwargs, artist_kws))
    return target.title


def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to bokeh for adding a label to the y axis."""
    kwargs = {"text_font_size": size, "text_color": color}
    target.yaxis.axis_label = string
    for key, value in _filter_kwargs(kwargs, artist_kws):
        setattr(target.yaxis, f"axis_label_{key}", value)


def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to bokeh for adding a label to the x axis."""
    kwargs = {"text_font_size": size, "text_color": color}
    target.xaxis.axis_label = string
    for key, value in _filter_kwargs(kwargs, artist_kws):
        setattr(target.xaxis, f"axis_label_{key}", value)


def ticks_size(value, target):  # pylint: disable=unused-argument
    """Interface to bokeh for setting ticks size."""
    warnings.warn("Setting ticks size not yet implemented in bokeh")


def remove_ticks(target, axis="y"):  # pylint: disable=unused-argument
    """Interface to bokeh for removing ticks from a plot."""
    warnings.warn("Setting ticks size not yet implemented in bokeh")


def remove_axis(target, axis="y"):
    """Interface to bokeh for removing axis from a plot."""
    if axis == "y":
        target.yaxis.visible = False
    elif axis == "x":
        target.yaxis.visible = False
    elif axis == "both":
        target.axis.visible = False
    else:
        raise ValueError(f"axis must be one of 'x', 'y' or 'both', got '{axis}'")
