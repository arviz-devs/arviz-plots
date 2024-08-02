# pylint: disable=unused-argument
"""None plotting backend.

Interface to no plotting backend, returned processed data without interfacing to any plotting
backend for actual drawing.

:term:`plots` are lists which are extended with kwarg dictionaries every time an artist
would be added to them.
"""
import warnings

import numpy as np

from .. import get_default_aes as get_agnostic_default_aes

ALLOW_KWARGS = True


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs):
    """Generate `n` default values for a given aesthetics keyword."""
    if aes_key not in kwargs:
        if aes_key in {"x", "y"}:
            return np.arange(n)
        return np.array([f"{aes_key}_{i}" for i in range(n)])
    return get_agnostic_default_aes(aes_key, n, kwargs)


def scale_fig_size(figsize, rows=1, cols=1, figsize_units=None):  # pylint: disable=unused-argument
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
        figsize_units = "dots"
    if figsize is None:
        width = 800
        height = 100 * (rows + 1) ** 1.1
        figsize_units = "dots"
    else:
        width, height = figsize
    if figsize_units == "inches":
        warnings.warn(
            f"Assuming dpi=100. Use figsize_units='dots' and figsize={figsize} "
            "to stop seeing this warning"
        )
        width *= 100
        height *= 100
    elif figsize_units != "dots":
        raise ValueError(f"figsize_units must be 'dots' or 'inches', but got {figsize_units}")

    return (width, height)


# object creation and i/o
def show(chart):
    """Show this :term:`chart`.

    Parameters
    ----------
    chart : chart type
    """
    raise TypeError("'none' backend objects can't be shown.")


def get_figsize(plot_collection):
    """Get the size of the :term:`chart` element and its units."""
    chart_element = plot_collection.viz["chart"].item()
    return chart_element["figsize"], chart_element["figsize_units"]


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
    """Create a :term:`chart` with a grid of :term:`plots` in it.

    Parameters
    ----------
    number : int
        Number of plots required
    rows, cols : int, default 1
        Number of rows and columns.
    figsize : (float, float), optional
        Size of the figure in `figsize_units`.
    figsize_units : {"inches", "dots"}, default "inches"
        Units in which `figsize` is given.
    squeeze : bool, default True
        Delete dimensions of size 1 in the resulting array of :term:`plots`
    sharex, sharey : bool, default False
        Flags that indicate the axis limits between the different plots should
        be shared.
    polar : bool, default False
    width_ratios : array_like of shape (cols,), optional
    plot_hspace : float, optional
    subplot_kws, **kwargs : mapping, optional
        Arguments passed downstream to the plotting backend.

    Returns
    -------
    chart : False
    plots : [] or ndarray of []
    """
    plots = np.empty((rows, cols), dtype=object)
    for i, idx in enumerate(np.ndindex((rows, cols))):
        plots[idx] = None if i + 1 > number else []
    if squeeze:
        plots = plots.squeeze()
    if not ALLOW_KWARGS:
        if subplot_kws:
            raise ValueError("'subplot_kws' is not empty")
        if kwargs:
            raise ValueError("kwargs are not empty")
    chart_element = {
        "figsize": figsize,
        "figsize_units": figsize_units,
        "sharex": sharex,
        "sharey": sharey,
        "polar": polar,
        "width_ratios": width_ratios,
        "plot_hspace": plot_hspace,
        "subplot_kws": subplot_kws,
        **kwargs,
    }
    return np.array(chart_element, dtype=object), plots


def _filter_kwargs(kwargs, artist_kws):
    """Filter a dictionary to remove all keys whose values are ``unset``."""
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    return {**artist_kws, **kwargs}


# "geoms"
def line(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to a line plot."""
    kwargs = {"color": color, "alpha": alpha, "width": width, "linestyle": linestyle}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {
        "function": "line",
        "x": np.atleast_1d(x),
        "y": np.atleast_1d(y),
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


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
    """Interface to a scatter plot."""
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
        "alpha": alpha,
        "facecolor": facecolor,
        "edgecolor": edgecolor,
        "width": width,
    }
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {
        "function": "scatter",
        "x": np.atleast_1d(x),
        "y": np.atleast_1d(y),
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


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
    """Interface to text annotation inside a plot."""
    kwargs = {
        "size": size,
        "alpha": alpha,
        "color": color,
        "vertical_align": vertical_align,
        "horizontal_align": horizontal_align,
    }
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {
        "function": "text",
        "x": np.atleast_1d(x),
        "y": np.atleast_1d(y),
        "string": string,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


def fill_between_y(x, y_bottom, y_top, target, *, color=unset, alpha=unset, **artist_kws):
    """Fill the region between y_bottom and y_top."""
    x = np.atleast_1d(x)
    y_bottom = np.atleast_1d(y_bottom)
    if y_bottom.size == 1:
        y_bottom = y_bottom.item()
    y_top = np.atleast_1d(y_top)
    if y_top.size == 1:
        y_top = y_top.item()
    kwargs = {"color": color, "alpha": alpha}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {
        "function": "fill_between_y",
        "x": x,
        "y_bottom": y_bottom,
        "y_top": y_top,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


# general plot appeareance
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to adding a title to a plot."""
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {"function": "title", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to adding a label to a plot's y axis."""
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {"function": "ylabel", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to adding a label to a plot's x axis."""
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {"function": "xlabel", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def xticks(ticks, labels, target, **artist_kws):
    """Interface to setting ticks and tick labels of the x axis."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {"function": "xticks", "ticks": ticks, "labels": labels, **artist_kws}
    target.append(artist_element)
    return artist_element


def yticks(ticks, labels, target, **artist_kws):
    """Interface to setting ticks and tick labels of the y axis."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {"function": "yticks", "ticks": ticks, "labels": labels, **artist_kws}
    target.append(artist_element)
    return artist_element


def xlim(lims, target, **artist_kws):
    """Interface to setting limits for the x axis."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {"function": "xlim", "lims": lims, **artist_kws}
    target.append(artist_element)
    return artist_element


def ticklabel_props(target, *, axis="both", size=unset, color=unset, **artist_kws):
    """Interface to setting size of tick labels."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    kwargs = {"axis": axis, "size": size, "color": color}
    artist_element = {"function": "ticklabel_props", **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def remove_ticks(target, *, axis="y"):
    """Interface to removing ticks from a plot."""
    artist_element = {
        "function": "remove_ticks",
        "axis": axis,
    }
    target.append(artist_element)
    return artist_element


def remove_axis(target, axis="y"):
    """Interface to removing axis from a plot."""
    artist_element = {
        "function": "remove_axis",
        "axis": axis,
    }
    target.append(artist_element)
    return artist_element


def legend(
    target,
    kwarg_list,
    label_list,
    title=None,  # pylint: disable=redefined-outer-name
    artist_type="line",
    artist_kwargs=None,
    **kwargs,
):
    """Interface to manually generated legends."""
    raise NotImplementedError("No legends in 'none' backend.")
