"""Matplotlib interface layer.

Notes
-----
Sets ``zorder`` of all non-text "geoms" to ``2`` so that elements plotted later
on are on top of previous ones.
"""

import warnings
from typing import Any, Dict

import numpy as np
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.pyplot import rcParams
from matplotlib.pyplot import show as _show
from matplotlib.pyplot import subplots
from matplotlib.text import Text

from ..none import get_default_aes as get_agnostic_default_aes
from .legend import legend


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs=None):
    """Generate `n` *matplotlib valid* default values for a given aesthetics keyword."""
    if kwargs is None:
        kwargs = {}
    if aes_key not in kwargs:
        default_prop_cycle = rcParams["axes.prop_cycle"].by_key()
        if ("color" in aes_key) or aes_key == "c":
            # fmt: off
            vals = [
                '#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6',
                '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd'
            ]
            # fmt: on
            vals = default_prop_cycle.get("color", vals)
        elif aes_key in {"linestyle", "ls"}:
            vals = ["-", "--", ":", "-."]
            vals = default_prop_cycle.get("linestyle", vals)
        elif aes_key in {"marker", "m"}:
            vals = ["o", "+", "^", "x", "d"]
            vals = default_prop_cycle.get("marker", vals)
        elif aes_key in default_prop_cycle:
            vals = default_prop_cycle[aes_key]
        else:
            return get_agnostic_default_aes(aes_key, n)
        return get_agnostic_default_aes(aes_key, n, {aes_key: vals})
    return get_agnostic_default_aes(aes_key, n, kwargs)


def scale_fig_size(figsize, rows=1, cols=1, figsize_units=None):
    """Scale figure properties according to figsize, rows and cols.

    Parameters
    ----------
    figsize : (float, float) or None
        Size of figure in `figsize_units`
    rows : int
        Number of rows
    cols : int
        Number of columns
    figsize_units : {"inches", "dots"}
        Ignored if `figsize` is ``None``

    Returns
    -------
    figsize : (float, float) or None
        Size of figure in dots
    labelsize : float
        fontsize for labels
    linewidth : float
        linewidth
    """
    if figsize_units is None:
        figsize_units = "inches"
    if figsize is None:
        default_width, default_height = rcParams["figure.figsize"]
        width = cols * (default_width if cols < 4 else 0.6 * default_width)
        height = default_height / 4 * (rows + 1) ** 1.1
        figsize_units = "inches"
    else:
        width, height = figsize
    if figsize_units == "inches":
        dpi = rcParams["figure.dpi"]
        width *= dpi
        height *= dpi
    elif figsize_units != "dots":
        raise ValueError(f"figsize_units must be 'dots' or 'inches', but got {figsize_units}")

    return (width, height)


# object creation and i/o
def show(chart):  # pylint: disable=unused-argument
    """Show all existing matplotlib figures."""
    _show()


def get_figsize(plot_collection):
    """Get the size of the :term:`chart` element and its units."""
    return plot_collection.viz["chart"].item().get_size_inches(), "inches"


def create_plotting_grid(
    number,
    rows=1,
    cols=1,
    figsize=None,
    figsize_units="inches",
    squeeze=True,
    sharex=False,
    sharey=False,
    polar=False,
    width_ratios=None,
    plot_hspace=None,
    subplot_kws=None,
    **kwargs,
):
    """Create a chart with a grid of plotting targets in it.

    Parameters
    ----------
    number : int
        Number of axes required
    rows, cols : int, default 1
        Number of rows and columns.
    figsize : (float, float), optional
        Size of the figure in `figsize_units`.
    figsize_units : {"inches", "dots"}, default "inches"
        Units in which `figsize` is given.
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
    if plot_hspace is not None:
        kwargs["gridspec_kw"] = kwargs.get("gridspec_kw", {}).copy()
        kwargs["gridspec_kw"].setdefault("wspace", plot_hspace)

    if figsize is not None:
        if figsize_units == "dots":
            dpi = rcParams["figure.dpi"]
            figsize = (figsize[0] / dpi, figsize[1] / dpi)
        elif figsize_units != "inches":
            raise ValueError(f"figsize_units must be 'dots' or 'inches', but got {figsize_units}")
    fig, axes = subplots(
        rows,
        cols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        width_ratios=width_ratios,
        figsize=figsize,
        subplot_kw=subplot_kws,
        **kwargs,
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
def hist(
    y,
    l_e,
    r_e,
    target,
    *,
    bottom=0,
    color=unset,
    alpha=unset,
    facecolor=unset,
    edgecolor=unset,
    **artist_kws,
):
    """Interface to matplotlib for a histogram bar plot."""
    artist_kws.setdefault("zorder", 2)
    widths = np.asarray(r_e) - np.asarray(l_e)
    if np.any(bottom != 0):
        height = y - bottom
    else:
        height = y
    if color is not unset:
        if facecolor is unset:
            facecolor = color
        if edgecolor is unset:
            edgecolor = color
    kwargs = {"bottom": bottom, "color": facecolor, "edgecolor": edgecolor, "alpha": alpha}
    return target.bar(
        l_e, height, width=widths, align="edge", **_filter_kwargs(kwargs, None, artist_kws)
    )


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
    fillable_marker = (marker is unset) or (marker in Line2D.filled_markers)
    if color is not unset:
        if facecolor is unset and edgecolor is unset:
            facecolor = color
            if fillable_marker:
                edgecolor = color
        elif facecolor is unset:
            facecolor = color
        elif edgecolor is unset and fillable_marker:
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


def errorbar(
    x,
    y,
    error,
    target,
    *,
    size=unset,
    marker=unset,
    color=unset,
    facecolor=unset,
    edgecolor=unset,
    width=unset,
    **artist_kws,
):
    """Interface to matplotlib for an errorbar plot."""
    artist_kws.setdefault("zorder", 2)
    fillable_marker = (marker is unset) or (marker in Line2D.filled_markers)
    if color is not unset:
        if facecolor is unset and edgecolor is unset:
            facecolor = color
            if fillable_marker:
                edgecolor = color
        elif facecolor is unset:
            facecolor = color
        elif edgecolor is unset and fillable_marker:
            edgecolor = color
    kwargs = {
        "capsize": size,
        "marker": marker,
        "markerfacecolor": facecolor,
        "markeredgecolor": edgecolor,
        "elinewidth": width,
    }
    return target.errorbar(x, y, error, **_filter_kwargs(kwargs, None, artist_kws))


def text(
    x,
    y,
    string,
    target,
    *,
    size=unset,
    alpha=unset,
    color=unset,
    vertical_align="center",
    horizontal_align="center",
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


def fill_between_y(x, y_bottom, y_top, target, **artist_kws):
    """Fill the area between y_bottom and y_top."""
    artist_kws.setdefault("linewidth", 0)
    return target.fill_between(x, y_bottom, y_top, **artist_kws)


# general plot appeareance
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a title to a plot."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_title(string, **_filter_kwargs(kwargs, Text, artist_kws))


def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a label to the y axis."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_ylabel(string, **_filter_kwargs(kwargs, Text, artist_kws))


def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a label to the x axis."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_xlabel(string, **_filter_kwargs(kwargs, Text, artist_kws))


def xticks(ticks, labels, target, **artist_kws):
    """Interface to matplotlib for adding x ticks and labels to a plot."""
    return target.set_xticks(ticks, labels, **artist_kws)


def yticks(ticks, labels, target, **artist_kws):
    """Interface to matplotlib for adding y ticks and labels to a plot."""
    return target.set_yticks(ticks, labels, **artist_kws)


def xlim(lims, target, **artist_kws):
    """Interface to matplotlib for setting limits for the x axis."""
    target.set_xlim(lims, **artist_kws)


def ticklabel_props(target, *, axis="both", size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for setting ticks size."""
    kwargs = {"labelsize": size, "labelcolor": color}
    target.tick_params(axis=axis, **_filter_kwargs(kwargs, None, artist_kws))


def remove_ticks(target, *, axis="y"):
    """Interface to matplotlib for removing ticks from a plot."""
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
