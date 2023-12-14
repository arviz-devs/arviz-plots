"""Matplotlib interface layer.

Notes
-----
Sets ``zorder`` of all non-text "geoms" to ``2`` so that elements plotted later
on are on top of previous ones.
"""
import warnings
from typing import Any, Dict

from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.pyplot import show as _show
from matplotlib.pyplot import subplots
from matplotlib.text import Text

__all__ = ["create_plotting_grid", "line", "scatter", "text", "title", "remove_axis"]


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()


# object creation and i/o
def show(chart):  # pylint: disable=unused-argument
    """Show all existing matplotlib figures."""
    _show()


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
        Passed to :func:`~matplotlib.pyplot.subplots` as ``subplot_kw``
    **kwargs: dict, optional
        Passed to :func:`~matplotlib.pyplot.subplots`

    Returns
    -------
    `~matplotlib.figure.Figure`
    `~matplotlib.axes.Axes` or ndarray of `~matplotlib.axes.Axes`
    """
    if subplot_kws is None:
        subplot_kws = {}
    subplot_kws = subplot_kws.copy()
    if polar:
        subplot_kws["projection"] = "polar"
    fig, axes = subplots(
        rows, cols, sharex=sharex, sharey=sharey, squeeze=squeeze, subplot_kw=subplot_kws, **kwargs
    )
    extra = (rows * cols) - number
    if extra > 0:
        for i, ax in enumerate(axes.ravel("C")):
            if i >= number:
                ax.set_axis_off()
    return fig, axes


# helper functions
def _filter_kwargs(kwargs, artist, artist_kws):
    """Filter a dictionary to remove all keys whose values are ``unset``.

    It also normalizes the matplotlib arguments and aliases to avoid clashing
    of aliases with their extended version.
    """
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    if artist is not None:
        artist_kws = normalize_kwargs(artist_kws.copy(), artist)
    return {**artist_kws, **kwargs}


# "geoms"
def line(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to matplotlib for a line plot."""
    artist_kws.setdefault("zorder", 2)
    kwargs = {"color": color, "alpha": alpha, "linewidth": width, "linestyle": linestyle}
    return target.plot(x, y, **_filter_kwargs(kwargs, Line2D, artist_kws))[0]


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
    """Interface to matplotlib for a scatter plot."""
    artist_kws.setdefault("zorder", 2)
    if color is not unset:
        if facecolor is not unset or edgecolor is not unset:
            warnings.warn(
                "color overrides facecolor and edgecolor. Their values will be ignored.",
                UserWarning,
            )
        facecolor = color
        edgecolor = color
    kwargs = {
        "s": size,
        "marker": marker,
        "alpha": alpha,
        "c": facecolor,
        "edgecolors": edgecolor,
        "linewidths": width,
    }
    return target.scatter(x, y, **_filter_kwargs(kwargs, None, artist_kws))


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
    """Interface to matplotlib for adding text to a plot."""
    kwargs = {
        "fontsize": size,
        "alpha": alpha,
        "color": color,
        "horizontalalignment": horizontal_align,
        "verticalalignment": vertical_align,
    }
    return target.text(x, y, string, **_filter_kwargs(kwargs, Text, artist_kws))


# general plot appeareance
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a title to a plot."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_title(string, **_filter_kwargs(kwargs, Text, artist_kws))

def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a title to a plot."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_ylabel(string, **_filter_kwargs(kwargs, Text, artist_kws))

def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a title to a plot."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_xlabel(string, **_filter_kwargs(kwargs, Text, artist_kws))

def remove_ticks(target, axis="y"):
    """Interface to matplotlib for removing axis from a plot."""
    if axis == "y":
        target.yaxis.set_ticks([])
    elif axis == "x":
        target.xaxis.set_ticks([])
    elif axis == "both":
        target.xaxis.set_ticks([])
        target.yaxis.set_ticks([])

def remove_axis(target, axis="y"):
    """Interface to matplotlib for removing axis from a plot."""
    target.spines["top"].set_visible(False)
    target.spines["right"].set_visible(False)
    target.tick_params(
        axis="both", which="both", left=axis == "x", top=False, right=False, bottom=axis == "y"
    )
    if axis == "y":
        target.yaxis.set_ticks([])
        target.spines["left"].set_visible(False)
        target.spines["bottom"].set_visible(True)
        target.xaxis.set_ticks_position("bottom")
        target.tick_params(axis="x", direction="out", width=1, length=3)
    elif axis == "x":
        target.xaxis.set_ticks([])
        target.spines["left"].set_visible(True)
        target.spines["bottom"].set_visible(False)
        target.xaxis.set_ticks_position("left")
        target.tick_params(axis="y", direction="out", width=1, length=3)
    elif axis == "both":
        target.xaxis.set_ticks([])
        target.yaxis.set_ticks([])
        target.spines["left"].set_visible(False)
        target.spines["bottom"].set_visible(False)
    else:
        raise ValueError(f"axis must be one of 'x', 'y' or 'both', got '{axis}'")
