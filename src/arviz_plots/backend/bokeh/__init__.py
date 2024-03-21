"""Bokeh interface layer."""
import warnings

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import Title
from bokeh.plotting import figure
from bokeh.plotting import show as _show

from .. import get_default_aes as get_agnostic_default_aes
from .legend import legend

__all__ = [
    "create_plotting_grid",
    "line",
    "scatter",
    "text",
    "title",
    "ylabel",
    "xlabel",
    "xticks",
    "yticks",
    "ticks_size",
    "remove_ticks",
    "remove_axis",
    "legend",
]


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs):
    """Generate `n` *bokeh valid* default values for a given aesthetics keyword."""
    if aes_key not in kwargs:
        if "color" in aes_key:
            # fmt: off
            vals = [
                '#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6',
                '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd'
            ]
            # fmt: on
        elif aes_key in {"linestyle", "line_dash"}:
            vals = ["solid", "dashed", "dotted", "dashdot"]
        elif aes_key == "marker":
            vals = ["circle", "cross", "triangle", "x", "diamond"]
        else:
            return get_agnostic_default_aes(aes_key, n, {})
        return get_agnostic_default_aes(aes_key, n, {aes_key: vals})
    return get_agnostic_default_aes(aes_key, n, kwargs)


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
    width_ratios=None,
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

    plot_widths = None
    if width_ratios is not None:
        if len(width_ratios) != cols:
            raise ValueError("width_ratios must be an iterable of length cols")
        plot_width = subplot_kws.get("width", 600)
        chart_width = plot_width * cols
        width_ratios = np.array(width_ratios, dtype=float)
        width_ratios /= width_ratios.sum()
        plot_widths = np.ceil(chart_width * width_ratios).astype(int)

    for row in range(rows):
        for col in range(cols):
            if width_ratios is not None:
                subplot_kws["width"] = plot_widths[col]
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
        if facecolor is unset and edgecolor is unset:
            facecolor = color
            edgecolor = color
        elif facecolor is unset:
            facecolor = color
        elif edgecolor is unset:
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
    kwargs = _filter_kwargs(kwargs, artist_kws)
    if marker == "|":
        kwargs["marker"] = "dash"
        kwargs["angle"] = np.pi / 2

    return target.scatter(np.atleast_1d(x), np.atleast_1d(y), **kwargs)


def text(
    x,
    y,
    string,
    target,
    *,
    size=unset,
    alpha=unset,
    color=unset,
    vertical_align="middle",
    horizontal_align="center",
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


def fill_between_y(x, y_bottom, y_top, target, **artist_kws):
    """Fill the area between y_bottom and y_top."""
    x = np.atleast_1d(x)
    y1 = np.atleast_1d(y_bottom)
    if y1.size == 1:
        y1 = y1.item()
    y2 = np.atleast_1d(y_top)
    if y2.size == 1:
        y2 = y2.item()
    return target.varea(x=x, y1=y1, y2=y2, **artist_kws)


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


def xticks(ticks, labels, target, **artist_kws):
    """Interface to bokeh for setting ticks and labels of the x axis."""
    target.xaxis.ticker = ticks
    if labels is not None:
        target.xaxis.major_label_overrides = {
            key.item() if hasattr(key, "item") else key: value for key, value in zip(ticks, labels)
        }
    for key, value in _filter_kwargs({}, artist_kws):
        setattr(target.xaxis, f"major_label_{key}", value)


def yticks(ticks, labels, target, **artist_kws):
    """Interface to bokeh for setting ticks and labels of the y axis."""
    target.yaxis.ticker = ticks
    if labels is not None:
        target.yaxis.major_label_overrides = {
            key.item() if hasattr(key, "item") else key: value for key, value in zip(ticks, labels)
        }
    for key, value in _filter_kwargs({}, artist_kws):
        setattr(target.yaxis, f"major_label_{key}", value)


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
