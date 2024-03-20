# pylint: disable=unused-argument
"""Common interface to plotting backends.

Each submodule within this module defines a common interface layer to different plotting libraries.

All other modules in ``arviz_subplots`` use this module to interact with the plotting
backends, never interacting directly. Thus, adding a new backend requires only
implementing this common interface for it, with no changes to any of the other modules.

Throughout the documentation of this module, there are a few type placeholders indicated
(e.g. ``chart type``). When implementing a module, these placeholders can be associated
to any type of the plotting backend or even custom objects, but all instances
of the same placeholder must use the same type (whatever that is).
"""
import numpy as np

error = NotImplementedError(
    "The `arviz_plots.backend` module itself is for reference only. "
    "A specific backend must be choosen, for example `arviz_plots.backend.bokeh` "
    "or `arviz_plots.backend.matplotlib`"
)


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs):
    """Generate `n` default values for a given aesthetics keyword."""
    if aes_key not in kwargs:
        if aes_key in {"x", "y"}:
            return np.arange(n)
        if aes_key == "alpha":
            return np.linspace(0.2, 0.7, n)
        return [None] * n
    aes_vals = kwargs[aes_key]
    n_aes_vals = len(aes_vals)
    if n_aes_vals >= n:
        return aes_vals[:n]
    return np.tile(aes_vals, (n // n_aes_vals) + 1)[:n]


# object creation and i/o
def show(chart):
    """Show this :term:`chart`.

    Parameters
    ----------
    chart : chart type
    """
    raise error


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
    """Create a :term:`chart` with a grid of :term:`plots` in it.

    Parameters
    ----------
    number : int
        Number of plots required
    rows, cols : int, default 1
        Number of rows and columns.
    squeeze : bool, default True
        Delete dimensions of size 1 in the resulting array of :term:`plots`
    sharex, sharey : bool, default False
        Flags that indicate the axis limits between the different plots should
        be shared.
    polar : bool, default False
    subplot_kws, **kwargs : mapping, optional
        Arguments passed downstream to the plotting backend.

    Returns
    -------
    chart : chart type
        The plotting backend object that represents the created :term:`chart`
    plots : plot type or ndarray of plot type
        An array of the plotting backend objects that represent the :term:`plots`.
        The returned object will be an array unless generating a 1x1 grid
        with `squeeze` set to True.
    """
    raise error


# "geoms"
def line(x, y, target, *, color=None, alpha=None, width=None, linestyle=None, **artist_kws):
    """Interface to a line plot.

    Add a line plot to the given `target`.

    Parameters
    ----------
    x, y : array-like
    target : plot type
    color : any
    alpha : float
    width : float
    linestyle : any
    **artist_kws : mapping
    """
    raise error


def scatter(
    x,
    y,
    target,
    *,
    size=None,
    marker=None,
    alpha=None,
    color=None,
    facecolor=None,
    edgecolor=None,
    width=None,
    **artist_kws,
):
    """Interface to a line plot.

    Add a line plot to the given `target`.

    Parameters
    ----------
    x, y : array-like
    target : plot type
    size : float or array-like of float
    marker : any
        The character ``|`` must be a valid marker as it is the default for rug plots.
    alpha : float
    color : any
        Set both facecolor and edgecolor simultaneously.
    facecolor : any
        Color of the marker filling.
    edgecolor : any
        Color of the marker edge.
    width : float
        Width of the marker edge.
    **artist_kws : mapping
    """
    raise error


def text(
    x,
    y,
    string,
    target,
    *,
    size=None,
    alpha=None,
    color=None,
    vertical_align=None,
    horizontal_align=None,
    **artist_kws,
):
    """Interface to text annotation inside a plot."""
    raise error


# general plot appeareance
def title(string, target, *, size=None, color=None, **artist_kws):
    """Interface to adding a title to a plot."""
    raise error


def ylabel(string, target, *, size=None, color=None, **artist_kws):
    """Interface to adding a label to a plot's y axis."""
    raise error


def xlabel(string, target, *, size=None, color=None, **artist_kws):
    """Interface to adding a label to a plot's x axis."""
    raise error


def ticks_size(value, target):
    """Interface to setting ticks size."""
    raise error


def remove_ticks(target, axis="y"):
    """Interface to removing ticks from a plot."""
    raise error


def remove_axis(target, axis="y"):
    """Interface to removing axis from a plot."""
    raise error


def legend(
    target,
    kwarg_list,
    label_list,
    title=None,  # pylint: disable=redefined-outer-name
    artist_type="line",
    artist_kwargs=None,
    **kwargs,
):
    """Interface to manually generated legends.

    Parameters
    ----------
    target : chart type
    kwarg_list : list of dict
        List of dictionaries that contain properties and their values for the miniatures
        in each entry of the legend.
    label_list : list of str
        List of labels of the entries in the legend.
    title : str, optional
        Title of the legend.
    artist_type : str, optional
        Type of the artist that will be used for the legend miniature.
    artist_kwargs : mapping, optional
        Keyword arguments passed to the miniatures artist.
    **kwargs : mapping, optional
        Keyword arguments passed to the backend legend generating function
    """
    raise error
