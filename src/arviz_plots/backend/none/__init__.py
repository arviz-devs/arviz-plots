# pylint: disable=unused-argument
"""None plotting backend.

Interface to no plotting backend, returned processed data without interfacing to any plotting
backend for actual drawing.

:term:`plots` are lists which are extended with kwarg dictionaries every time an visual
would be added to them.
"""
import warnings

import numpy as np

from .legend import legend

ALLOW_KWARGS = True


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs=None):
    """Generate `n` default values for a given aesthetics keyword.

    Parameters
    ----------
    aes_key : str
        The key for which default values should be generated.
        Ideally part of {ref}`common interface arguments <backend_interface_arguments>`.
    n : int
        Number of values to generate.
    kwargs : mapping of {str : array_like}, optional
        Mapping with aesthetic keywords as keys and its correponding values as values.
        If `aes_key` is present, the provided values will be used, repeating them
        with :func:`numpy.tile` if necessary.

    Returns
    -------
    ndarray of shape (n,)
        The requested `n` default values for `aes_key`. They might not be unique.
    """
    if kwargs is None:
        kwargs = {}
    if aes_key not in kwargs:
        if aes_key in {"x", "y"}:
            return np.arange(n)
        if aes_key == "alpha":
            return np.linspace(0.2, 0.7, n)
        return np.array([f"{aes_key}_{i}" for i in range(n)])
    aes_vals = kwargs[aes_key]
    n_aes_vals = len(aes_vals)
    if n_aes_vals >= n:
        return aes_vals[:n]
    return np.tile(aes_vals, (n // n_aes_vals) + 1)[:n]


def scale_fig_size(figsize, rows=1, cols=1, figsize_units=None):
    """Scale figure properties according to figsize, rows and cols.

    Provide a default figure size given `rows` and `cols`.

    Parameters
    ----------
    figsize : tuple of (float, float) or None
        Size of figure in `figsize_units`
    rows : int, default 1
        Number of rows
    cols : int, default 1
        Number of columns
    figsize_units : {"inches", "dots"}, optional
        Ignored if `figsize` is ``None``

    Returns
    -------
    figsize : tuple of (float, float) or None
        Size of figure in dots
    """
    if figsize_units is None:
        figsize_units = "dots"
    if figsize is None:
        width = cols * (400 if cols < 4 else 250)
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
def show(figure):
    """Show this :term:`figure`.

    Parameters
    ----------
    figure : figure_type
    """
    raise TypeError("'none' backend objects can't be shown.")


def savefig(figure, path, **kwargs):
    """Show this :term:`figure`.

    Parameters
    ----------
    figure : figure_type
        The figure to save.
    path : pathlib.Path
        The path to save the figure to.
    **kwargs : dict, optional
        Additional keyword arguments.
    """
    raise TypeError("'none' backend figures can't be saved.")


def get_figsize(plot_collection):
    """Get the size of the :term:`figure` element and its units."""
    figure_element = plot_collection.viz["figure"].item()
    return figure_element["figsize"], figure_element["figsize_units"]


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
    """Create a :term:`figure` with a grid of :term:`plots` in it.

    Parameters
    ----------
    number : int
        Number of plots required
    rows, cols : int, default 1
        Number of rows and columns.
    figsize : tuple of (float, float), optional
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
    figure : False
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
    figure_element = {
        "figsize": figsize,
        "figsize_units": figsize_units,
        "sharex": sharex,
        "sharey": sharey,
        "polar": polar,
        "width_ratios": width_ratios,
        "height_ratios": height_ratios,
        "plot_hspace": plot_hspace,
        "subplot_kws": subplot_kws,
        **kwargs,
    }
    return np.array(figure_element, dtype=object), plots


def _filter_kwargs(kwargs, artist_kws):
    """Filter a dictionary to remove all keys whose values are ``unset``."""
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    return {**artist_kws, **kwargs}


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
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    if color is not unset:
        if facecolor is unset:
            facecolor = color
        if edgecolor is unset:
            edgecolor = color
    kwargs = {"bottom": bottom, "facecolor": facecolor, "edgecolor": edgecolor}
    artist_element = {
        "function": "hist",
        "l_e": np.atleast_1d(l_e),
        "r_e": np.atleast_1d(r_e),
        "y": np.atleast_1d(y),
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


# "geoms"
def line(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to a line plot."""
    kwargs = {"color": color, "alpha": alpha, "width": width, "linestyle": linestyle}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "line",
        "x": np.atleast_1d(x),
        "y": np.atleast_1d(y),
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


def multiple_lines(
    x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws
):
    """Interface to multiple lines.

    Parameters
    ----------
    x : (N,) array-like
        Shared data for the x axis
    y : (N, M) array-like
        Data for the y values of the multiple lines
    target : list[Any]
    color, alpha, width, linestyle : Any, optional
        See {ref}`backend_interface_arguments` for their description
    **artist_kws
        Extra keyword arguments. For the none backend they are stored as is
        to allow for rendering of the plot later on.

    Returns
    -------
    dict
        dictionary with the provided arguments and a "function" key to identify
        the backend function.
    """
    kwargs = {"color": color, "alpha": alpha, "width": width, "linestyle": linestyle}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "multiple_lines",
        "x": x,
        "y": y,
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
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "scatter",
        "x": np.atleast_1d(x),
        "y": np.atleast_1d(y),
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


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
    """Interface to a step line."""
    kwargs = {
        "color": color,
        "alpha": alpha,
        "width": width,
        "linestyle": linestyle,
        "step_mode": step_mode,
    }
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "step",
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
        raise ValueError(f"artist_kws not empty: {artist_kws}")
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
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "fill_between_y",
        "x": x,
        "y_bottom": y_bottom,
        "y_top": y_top,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


def vline(x, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to a vertical line spanning the whole axes."""
    kwargs = {"color": color, "alpha": alpha, "width": width, "linestyle": linestyle}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "vline",
        "x": x,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


def hline(y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to a horizontal line spanning the whole axes."""
    kwargs = {"color": color, "alpha": alpha, "width": width, "linestyle": linestyle}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "vline",
        "y": y,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


def vspan(xmin, xmax, target, *, color=unset, alpha=unset, **artist_kws):
    """Interface to a vertical shaded region spanning the whole axes."""
    kwargs = {"color": color, "alpha": alpha}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {
        "function": "vspan",
        "xmin": xmin,
        "xmax": xmax,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


def hspan(ymin, ymax, target, *, color=unset, alpha=unset, **artist_kws):
    """Interface to a horizontal shaded region spanning the whole axes."""
    kwargs = {"color": color, "alpha": alpha}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError("artist_kws not empty")
    artist_element = {
        "function": "hspan",
        "ymin": ymin,
        "ymax": ymax,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


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
    """Interface to a line from y_bottom to y_top at given value of x."""
    kwargs = {"color": color, "alpha": alpha, "width": width, "linestyle": linestyle}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "ciliney",
        "x": np.atleast_1d(x),
        "y_bottom": np.atleast_1d(y_bottom),
        "y_top": np.atleast_1d(y_top),
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


# general plot appeareance
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to adding a title to a plot."""
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "title", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to adding a label to a plot's y axis."""
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "ylabel", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to adding a label to a plot's x axis."""
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "xlabel", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def xticks(ticks, labels, target, *, rotation=unset, **artist_kws):
    """Interface to setting ticks and tick labels of the x axis."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "xticks",
        "ticks": ticks,
        "labels": labels,
        "rotation": rotation,
        **artist_kws,
    }
    target.append(artist_element)
    return artist_element


def yticks(ticks, labels, target, *, rotation=unset, **artist_kws):
    """Interface to setting ticks and tick labels of the y axis."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "yticks",
        "ticks": ticks,
        "labels": labels,
        "rotation": rotation,
        **artist_kws,
    }
    target.append(artist_element)
    return artist_element


def set_ticklabel_visibility(target, *, axis="both", visible=True):
    """Interface to setting visibility of tick labels."""
    artist_element = {
        "function": "set_ticklabel_visibility",
        "axis": axis,
        "visible": visible,
    }
    target.append(artist_element)
    return artist_element


def xlim(lims, target, **artist_kws):
    """Interface to setting limits for the x axis."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "xlim", "lims": lims, **artist_kws}
    target.append(artist_element)
    return artist_element


def ylim(lims, target, **artist_kws):
    """Interface to setting limits for the y axis."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "ylim", "lims": lims, **artist_kws}
    target.append(artist_element)
    return artist_element


def ticklabel_props(target, *, axis="both", size=unset, color=unset, **artist_kws):
    """Interface to setting size of tick labels."""
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
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


def set_y_scale(target, scale):
    """Interface to setting the y scale of a plot."""
    artist_element = {
        "function": "set_y_scale",
        "scale": scale,
    }
    target.append(artist_element)
    return artist_element


def grid(target, axis, color):
    """Interface to setting a grid in any axis."""
    artist_element = {
        "function": "grid",
        "axis": axis,
        "color": color,
    }
    target.append(artist_element)
    return artist_element
