# pylint: disable=no-self-use
"""Matplotlib interface layer.

Notes
-----
Sets ``zorder`` of all non-text "geoms" to ``2`` so that elements plotted later
on are on top of previous ones.
"""

import warnings

import matplotlib.colors as mcolors
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import ticker
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.pyplot import rcParams
from matplotlib.pyplot import show as _show
from matplotlib.pyplot import subplots
from matplotlib.text import Text

from ..alias_utils import create_aesthetic_handlers
from ..none import get_default_aes as get_agnostic_default_aes


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()


class SquareRootScale(mscale.ScaleBase):
    """ScaleBase class for generating square root scale."""

    name = "sqrt"

    def __init__(self, axis, **kwargs):  # pylint: disable=unused-argument
        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        """Set the locators and formatters to default."""
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):  # pylint: disable=unused-argument
        """Limit the range of the scale."""
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        """Square root transformation."""

        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, values):
            """Transform the data."""
            return np.array(values) ** 0.5

        def inverted(self):
            """Invert the transformation."""
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        """Inverted square root transformation."""

        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, values):
            """Transform the data."""
            return np.array(values) ** 2

        def inverted(self):
            """Invert the transformation."""
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        """Get the transformation."""
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)


def get_background_color():
    """Get the background color."""
    bg_color = rcParams["figure.facecolor"]
    try:
        bg_color = mcolors.to_hex(bg_color)
    except ValueError:
        warnings.warn(
            "The background color is not a valid matplotlib color. "
            "Returning the default value '#ffffff'."
        )
        bg_color = "#ffffff"
    return bg_color


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs=None):
    """Generate `n` *matplotlib valid* default values for a given aesthetics keyword."""
    if kwargs is None:
        kwargs = {}
    if aes_key not in kwargs:
        default_prop_cycle = rcParams["axes.prop_cycle"].by_key()
        if "color" in aes_key:
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
            vals = ["o", "+", "^", "x", "d", "s", "."]
            vals = default_prop_cycle.get("marker", vals)
        elif aes_key in default_prop_cycle:
            vals = default_prop_cycle[aes_key]
        else:
            return get_agnostic_default_aes(aes_key, n)
        return get_agnostic_default_aes(aes_key, n, {aes_key: vals})
    return get_agnostic_default_aes(aes_key, n, kwargs)


# Create aesthetic alias handling functions using the factory
expand_aesthetic_aliases = create_aesthetic_handlers(get_default_aes, get_background_color)


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
def show(figure):  # pylint: disable=unused-argument
    """Show all existing matplotlib figures."""
    _show()


def savefig(figure, path, **kwargs):
    """Save the figure to a file.

    Parameters
    ----------
    figure : `~matplotlib.figure.Figure`
        The figure to save.
    path : pathlib.Path
        Path to the file where the figure will be saved.
    **kwargs : dict, optional
        Additional keyword arguments passed to `matplotlib.pyplot.savefig`.
    """
    figure.savefig(path, **kwargs)


def get_figsize(plot_collection):
    """Get the size of the :term:`figure` element and its units."""
    return plot_collection.viz["figure"].item().get_size_inches(), "inches"


def create_plotting_grid(
    number,
    rows=1,
    cols=1,
    *,
    figsize=None,
    figsize_units="inches",
    squeeze=True,
    sharex=False,
    sharey=False,
    polar=False,
    width_ratios=None,
    height_ratios=None,
    plot_hspace=None,
    subplot_kws=None,
    **kwargs,
):
    """Create a figure with a grid of plotting targets in it.

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
        height_ratios=height_ratios,
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
def _filter_kwargs(kwargs, visual, artist_kws):
    """Filter a dictionary to remove all keys whose values are ``unset``.

    It also normalizes the matplotlib arguments and aliases to avoid clashing
    of aliases with their extended version.
    """
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    if visual is not None:
        artist_kws = normalize_kwargs(artist_kws.copy(), visual)
    return {**artist_kws, **kwargs}


# "geoms"
@expand_aesthetic_aliases
def hist(
    y,
    l_e,
    r_e,
    target,
    *,
    bottom=0,
    color=unset,
    facecolor=unset,
    edgecolor=unset,
    alpha=unset,
    **artist_kws,
):
    """Interface to matplotlib for a histogram bar plot."""
    artist_kws.setdefault("zorder", 2)
    if np.any(bottom != 0):
        height = y - bottom
    else:
        height = y
    if color is not unset:
        if facecolor is unset:
            facecolor = color
        if edgecolor is unset:
            edgecolor = color

    kwargs = {"color": facecolor, "edgecolor": edgecolor, "alpha": alpha}
    return target.fill_between(
        np.r_[l_e, r_e[-1]],
        np.r_[height, height[-1]],
        step="post",
        **_filter_kwargs(kwargs, None, artist_kws),
    )


@expand_aesthetic_aliases
def line(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to matplotlib for a line plot."""
    artist_kws.setdefault("zorder", 2)
    kwargs = {"color": color, "alpha": alpha, "linewidth": width, "linestyle": linestyle}
    return target.plot(x, y, **_filter_kwargs(kwargs, Line2D, artist_kws))[0]


@expand_aesthetic_aliases
def multiple_lines(
    x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws
):
    """Interface to matplotlib for a multiple line plot using a single LineCollection."""
    artist_kws.setdefault("zorder", 2)
    y_2d = np.atleast_2d(y)
    segments = [np.column_stack([x, y_col]) for y_col in y_2d.T]
    plot_kwargs = {"colors": color, "alpha": alpha, "linewidths": width, "linestyles": linestyle}
    filtered_kwargs = _filter_kwargs(plot_kwargs, LineCollection, artist_kws)
    line_collection = LineCollection(segments, **filtered_kwargs)
    target.add_collection(line_collection)

    target.autoscale_view()

    return line_collection


@expand_aesthetic_aliases
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


@expand_aesthetic_aliases
def step(
    x,
    y,
    target,
    *,
    color=unset,
    alpha=unset,
    width=unset,
    linestyle=unset,
    step_mode=unset,
    **artist_kws,
):
    """Interface to matplotlib for a step line."""
    artist_kws.setdefault("zorder", 2)
    kwargs = {"color": color, "alpha": alpha, "linewidth": width, "linestyle": linestyle}
    if step_mode is not unset:
        if step_mode == "before":
            kwargs["where"] = "pre"
        elif step_mode == "after":
            kwargs["where"] = "post"
        else:
            kwargs["where"] = "mid"
    return target.step(x, y, **_filter_kwargs(kwargs, Line2D, artist_kws))[0]


@expand_aesthetic_aliases
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


@expand_aesthetic_aliases
def fill_between_y(x, y_bottom, y_top, target, **artist_kws):
    """Fill the area between y_bottom and y_top."""
    artist_kws.setdefault("linewidth", 0)
    return target.fill_between(x, y_bottom, y_top, **artist_kws)


@expand_aesthetic_aliases
def vline(x, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to matplotlib for a vertical line spanning the whole axes."""
    artist_kws.setdefault("zorder", 0)
    kwargs = {"color": color, "alpha": alpha, "linewidth": width, "linestyle": linestyle}
    return target.axvline(x, **_filter_kwargs(kwargs, Line2D, artist_kws))


@expand_aesthetic_aliases
def hline(y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to matplotlib for a horizontal line spanning the whole axes."""
    artist_kws.setdefault("zorder", 0)
    kwargs = {"color": color, "alpha": alpha, "linewidth": width, "linestyle": linestyle}
    return target.axhline(y, **_filter_kwargs(kwargs, Line2D, artist_kws))


@expand_aesthetic_aliases
def vspan(xmin, xmax, target, *, color=unset, alpha=unset, **artist_kws):
    """Interface to matplotlib for a vertical shaded region spanning the whole axes."""
    artist_kws.setdefault("zorder", 0)
    kwargs = {"color": color, "alpha": alpha}
    return target.axvspan(xmin, xmax, **_filter_kwargs(kwargs, None, artist_kws))


@expand_aesthetic_aliases
def hspan(ymin, y_max, target, *, color=unset, alpha=unset, **artist_kws):
    """Interface to matplotlib for a horizontal shaded region spanning the whole axes."""
    artist_kws.setdefault("zorder", 0)
    kwargs = {"color": color, "alpha": alpha}
    return target.axhspan(ymin, y_max, **_filter_kwargs(kwargs, None, artist_kws))


@expand_aesthetic_aliases
def ciliney(
    x,
    y_bottom,
    y_top,
    target,
    *,
    color=unset,
    alpha=unset,
    width=unset,
    linestyle=unset,
    **artist_kws,
):
    """Interface to matplotlib for a line from y_bottom to y_top at given value of x."""
    artist_kws.setdefault("zorder", 2)
    kwargs = {"color": color, "alpha": alpha, "linewidth": width, "linestyle": linestyle}
    return target.plot([x, x], [y_bottom, y_top], **_filter_kwargs(kwargs, Line2D, artist_kws))[0]


# general plot appeareance
@expand_aesthetic_aliases
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a title to a plot."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_title(string, **_filter_kwargs(kwargs, Text, artist_kws))


@expand_aesthetic_aliases
def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a label to the y axis."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_ylabel(string, **_filter_kwargs(kwargs, Text, artist_kws))


@expand_aesthetic_aliases
def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to matplotlib for adding a label to the x axis."""
    kwargs = {"fontsize": size, "color": color}
    return target.set_xlabel(string, **_filter_kwargs(kwargs, Text, artist_kws))


def xticks(ticks, labels, target, *, rotation=unset, **artist_kws):
    """Interface to matplotlib for adding x ticks and labels to a plot."""
    if rotation is not unset:
        artist_kws["rotation"] = rotation
    return target.set_xticks(ticks, labels, **artist_kws)


def yticks(ticks, labels, target, *, rotation=unset, **artist_kws):
    """Interface to matplotlib for adding y ticks and labels to a plot."""
    if rotation is not unset:
        artist_kws["rotation"] = rotation
    return target.set_yticks(ticks, labels, **artist_kws)


def set_ticklabel_visibility(target, *, axis="both", visible=True):
    """Interface to matplotlib for setting visibility of tick labels."""
    if axis == "both":
        target.tick_params(axis="both", labelbottom=visible, labelleft=visible)
    elif axis == "x":
        target.tick_params(axis="x", labelbottom=visible)
    elif axis == "y":
        target.tick_params(axis="y", labelleft=visible)
    else:
        raise ValueError(f"axis must be one of 'x', 'y' or 'both', got '{axis}'")


def xlim(lims, target, **artist_kws):
    """Interface to matplotlib for setting limits for the x axis."""
    target.set_xlim(lims, **artist_kws)


def ylim(lims, target, **artist_kws):
    """Interface to matplotlib for setting limits for the y axis."""
    target.set_ylim(lims, **artist_kws)


@expand_aesthetic_aliases
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
        target.xaxis.set_ticks_position("bottom")
        target.tick_params(axis="y", direction="out", width=1, length=3)
    elif axis == "both":
        target.set_axis_off()
    else:
        raise ValueError(f"axis must be one of 'x', 'y' or 'both', got '{axis}'")


def set_y_scale(target, scale):
    """Interface to matplotlib for setting the y scale of a plot."""
    target.set_yscale(scale)


@expand_aesthetic_aliases
def grid(target, axis, color):
    """Interface to matplotlib for setting a grid in any axis."""
    target.grid(axis=axis, color=color)
