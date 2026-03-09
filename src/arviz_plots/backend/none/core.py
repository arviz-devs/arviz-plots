# pylint: disable=unused-argument
"""None plotting backend.

Interface to no plotting backend, returned processed data without interfacing to any plotting
backend for actual drawing.

:term:`plots` are lists which are extended with kwarg dictionaries every time a visual
would be added to them.
"""
import warnings

import numpy as np

ALLOW_KWARGS = True


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""

    def __repr__(self):
        """Set custom repr for docs."""
        return "<unset>"


unset = UnsetDefault()


def get_background_color():
    """Get the background color of active style.

    See Also
    --------
    arviz_plots.backend.bokeh.get_background_color
    arviz_plots.backend.matplotlib.get_background_color
    arviz_plots.backend.plotly.get_background_color
    """
    return "#ffffff"


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs=None):
    """Generate `n` default values for a given aesthetics keyword.

    Parameters
    ----------
    aes_key : str
        The key for which default values should be generated.
        Note :term:`aesthetics` can be arbitrary keyword arguments, but whenever
        possible you should use
        :ref:`common interface arguments <backend_interface_arguments>`
        to take advantage of all the available features and defaults.
        For example, `color` will get a default cycle assigned automatically
        whereas `fill_color` won't.
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

    See Also
    --------
    arviz_plots.backend.bokeh.get_default_aes
    arviz_plots.backend.matplotlib.get_default_aes
    arviz_plots.backend.plotly.get_default_aes
    """
    if kwargs is None:
        kwargs = {}
    if aes_key not in kwargs:
        if aes_key in {"x", "y"}:
            return np.arange(n)
        if aes_key == "alpha":
            return np.linspace(0.2, 0.7, n)
        return np.array([f"C{i}" for i in range(n)])
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
    figsize : tuple of (float, float)
        Size of figure in dots

    See Also
    --------
    arviz_plots.backend.bokeh.scale_fig_size
    arviz_plots.backend.matplotlib.scale_fig_size
    arviz_plots.backend.plotly.scale_fig_size
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

    See Also
    --------
    arviz_plots.backend.bokeh.show
    arviz_plots.backend.matplotlib.show
    arviz_plots.backend.plotly.show
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

    See Also
    --------
    arviz_plots.backend.bokeh.savefig
    arviz_plots.backend.matplotlib.savefig
    arviz_plots.backend.plotly.savefig
    """
    raise TypeError("'none' backend figures can't be saved.")


def set_figure_title(figure, string, *, color=unset, size=unset, **artist_kws):
    """Set a title for the entire figure.

    Parameters
    ----------
    figure : dict
        The figure element dict.
    string : str
        The title text.
    color : optional
        Color of the title text.
    size : optional
        Font size of the title.
    **artist_kws
        Passed to the backend title-setting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.figure.Figure.suptitle`
        * plotly -> :meth:`~plotly.graph_objects.Figure.update_layout` (``title=...``)
        * bokeh -> :attr:`~bokeh.plotting.figure.Figure.title`

    Returns
    -------
    dict
        The figure element dict (unchanged).
    dict
        The title element dict.
    """
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    kwargs = {"color": color, "size": size}
    title_element = {
        "function": "set_figure_title",
        "string": string,
        **_filter_kwargs(kwargs, artist_kws),
    }
    return figure, title_element


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
    if squeeze and rows * cols == 1:
        plots = []
    elif squeeze:
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
    """Filter keyword arguments removing values set to ``unset``.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments where some values may be ``unset``.
        Keys whose values are ``unset`` are removed.
    artist_kws : dict
        Additional keyword arguments provided by the user.

    Returns
    -------
    dict
        Dictionary combining ``artist_kws`` with the filtered ``kwargs``.
    """
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
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
    facecolor=unset,
    edgecolor=unset,
    alpha=unset,
    **artist_kws,
):
    """Interface to a histogram plot.

    Parameters
    ----------
    y : array-like of shape (n,)
        Heights of the histogram bars corresponding to each bin.
    l_e, r_e : array-like of shape (n,)
        Left and right edges of the histogram bins.
    target : PlotObject
        The backend object representing a :term:`plot` where this :term:`visual`
        should be added.
    bottom : float, default 0
        Baseline from which the bars are drawn.
    color, facecolor, edgecolor, alpha : any
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.bar`
        * plotly -> :class:`~plotly.graph_objects.Bar`
        * bokeh -> :meth:`~bokeh.plotting.figure.quad`

    Returns
    -------
    hist_visual : any
        The backend object representing the generated collection of histogram bars.
    """
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


def line(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to a line plot.

    Parameters
    ----------
    x, y : array-like of shape (n,)
        The x and y data to be plotted as a line
    target : PlotObject
        The backend object representing a :term:`plot` where this :term:`visual` should be added.
    color, alpha, width, linestyle
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.plot`
        * plotly -> :class:`~plotly.graph_objects.Scatter` with (``mode="lines"``)
        * bokeh -> :meth:`~bokeh.plotting.figure.line`

    Returns
    -------
    line_visual : any
        The backend object representing the generated line.
    """
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
    """Interface to multiple line plots.

    Parameters
    ----------
    x : array-like of shape (n,)
        Shared x-axis data.
    y : array-like of shape (n, m)
        Y data defining multiple lines.
    target : PlotObject
        The backend object representing a :term:`plot` where these
        :term:`visual` elements should be added.
    color, alpha, width, linestyle : any, optional
        Properties of the generated :term:`visual` elements.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.plot`
        * plotly -> :class:`~plotly.graph_objects.Scatter` with (``mode="lines"``)
        * bokeh -> :meth:`~bokeh.plotting.figure.line`

    Returns
    -------
    multiple_lines_visual : any
        The backend object representing the generated collection of lines.
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
    """Interface to a scatter plot.

    Parameters
    ----------
    x, y : array_like of shape (n,)
        Data for the points to plot
    target : PlotObject
        The backend object representing a :term:`plot` where this :term:`visual` should be added.
    size, marker, alpha, color, facecolor, edgecolor, width : any
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.scatter`
        * plotly -> :class:`~plotly.graph_objects.Scatter` with (``mode="markers"``)
        * bokeh -> :meth:`~bokeh.plotting.figure.scatter`

    Returns
    -------
    scatter_visual : any
        The backend object representing the plotted collection of points.
    """
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
    """Interface to a step line plot.

    Parameters
    ----------
    x, y : array-like of shape (n,)
        Data defining the step line.
    target : PlotObject
        The backend object representing a :term:`plot` where this :term:`visual`
        should be added.
    color, alpha, width, linestyle : any
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    step_mode : any, optional
        Defines how the step transitions are drawn (e.g. ``pre``, ``post``, ``mid``).
        Interpretation depends on the plotting backend.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.step`
        * plotly -> :class:`~plotly.graph_objects.Scatter` (``line_shape="hv"`` or similar)
        * bokeh -> :meth:`~bokeh.plotting.figure.step`

    Returns
    -------
    step_visual : any
        The backend object representing the generated step line.
    """
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
    """Interface to a text annotation inside a plot.

    Parameters
    ----------
    x, y : float or array-like
        Coordinates where the text should be placed.
    string : str
        The text content to display.
    target : PlotObject
        The backend object representing a :term:`plot` where this :term:`visual`
        should be added.
    size, alpha, color : any, optional
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    vertical_align, horizontal_align : any, optional
        Alignment properties of the text.
        Interpretation depends on the plotting backend.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.text`
        * plotly -> :class:`~plotly.graph_objects.layout.Annotation`
        * bokeh -> :class:`~bokeh.models.annotations.Label`

    Returns
    -------
    text_visual : any
        The backend object representing the generated text annotation.
    """
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
    """Fill the region between two y-values.

    Parameters
    ----------
    x : array-like of shape (n,)
        X coordinates of the filled region.
    y_bottom, y_top : array-like or scalar
        Lower and upper y-values defining the region to fill.
    target : PlotObject
        The backend object representing a :term:`plot` where this :term:`visual`
        should be added.
    color, alpha : any, optional
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.fill_between`
        * plotly -> :class:`~plotly.graph_objects.Scatter`
        * bokeh -> :meth:`~bokeh.plotting.figure.varea`

    Returns
    -------
    fill_between_visual : any
        The backend object representing the generated filled region.
    """
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
    """Interface to a vertical reference line.

    Parameters
    ----------
    x : float or array-like
        X position(s) where the vertical line(s) should be drawn.
    target : PlotObject
        The backend object representing a :term:`plot` where this :term:`visual`
        should be added.
    color, alpha, width, linestyle : any, optional
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.axvline`
        * plotly -> :class:`~plotly.graph_objects.layout.Shape`
        * bokeh -> :class:`~bokeh.models.Span`

    Returns
    -------
    vline_visual : any
        The backend object representing the generated vertical line.
    """
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
    """Interface to a horizontal reference line.

    Parameters
    ----------
    y : float or array-like
        Y position(s) where the horizontal line(s) should be drawn.
    target : PlotObject
        The backend object representing a :term:`plot` where this :term:`visual`
        should be added.
    color, alpha, width, linestyle : any, optional
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.axhline`
        * plotly -> :class:`~plotly.graph_objects.layout.Shape`
        * bokeh -> :class:`~bokeh.models.Span`

    Returns
    -------
    hline_visual : any
        The backend object representing the generated horizontal line.
    """
    kwargs = {"color": color, "alpha": alpha, "width": width, "linestyle": linestyle}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "hline",
        "y": y,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


def vspan(xmin, xmax, target, *, color=unset, alpha=unset, **artist_kws):
    """Interface to a vertical shaded region.

    Parameters
    ----------
    xmin, xmax : float
        The start and end x-values of the shaded region.
    target : PlotObject
        The backend object representing a :term:`plot` where this
        :term:`visual` should be added.
    color, alpha : any, optional
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.axvspan`
        * plotly -> :class:`~plotly.graph_objects.layout.Shape`
        * bokeh -> :meth:`~bokeh.plotting.figure.varea`

    Returns
    -------
    vspan_visual : any
        The backend object representing the generated shaded region.
    """
    kwargs = {"color": color, "alpha": alpha}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {
        "function": "vspan",
        "xmin": xmin,
        "xmax": xmax,
        **_filter_kwargs(kwargs, artist_kws),
    }
    target.append(artist_element)
    return artist_element


def hspan(ymin, ymax, target, *, color=unset, alpha=unset, **artist_kws):
    """Interface to a horizontal shaded region.

    Parameters
    ----------
    ymin, ymax : float
        The start and end y-values of the shaded region.
    target : PlotObject
        The backend object representing a :term:`plot` where this
        :term:`visual` should be added.
    color, alpha : any, optional
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.axhspan`
        * plotly -> :class:`~plotly.graph_objects.layout.Shape`
        * bokeh -> :meth:`~bokeh.plotting.figure.harea`

    Returns
    -------
    hspan_visual : any
        The backend object representing the generated shaded region.
    """
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
    """Interface to a vertical interval line at a given x position.

    Parameters
    ----------
    x : float or array-like
        X position(s) where the interval line(s) should be drawn.
    y_bottom, y_top : array-like
        Lower and upper y-values defining the interval.
    target : PlotObject
        The backend object representing a :term:`plot` where this
        :term:`visual` should be added.
    color, alpha, width, linestyle : any, optional
        Properties of the generated :term:`visual`.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.vlines`
        * plotly -> :class:`~plotly.graph_objects.Scatter`
        * bokeh -> :meth:`~bokeh.plotting.figure.segment`

    Returns
    -------
    ciliney_visual : any
        The backend object representing the generated interval line.
    """
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


# general plot appearance
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to a title.

    Parameters
    ----------
    string : str
        Text to use as the title.
    target : PlotObject
        The backend object representing a :term:`plot` where this
        :term:`visual` should be added.
    size : any, optional
        Size of the title text.
    color : any, optional
        Color of the title text.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.set_title`
        * plotly -> :class:`~plotly.graph_objects.layout.Title`
        * bokeh -> :class:`~bokeh.models.Title`

    Returns
    -------
    title_visual : any
        The backend object representing the generated title.
    """
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "title", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to a y-axis label.

    Parameters
    ----------
    string : str
        Text to use as the y-axis label.
    target : PlotObject
        The backend object representing a :term:`plot` where this
        :term:`visual` should be added.
    size : any, optional
        Size of the label text.
    color : any, optional
        Color of the label text.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.set_ylabel`
        * plotly -> :class:`~plotly.graph_objects.layout.YAxis`
        * bokeh -> :class:`~bokeh.models.Axis`

    Returns
    -------
    ylabel_visual : any
        The backend object representing the generated y-axis label.
    """
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "ylabel", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to an x-axis label.

    Parameters
    ----------
    string : str
        Text to use as the x-axis label.
    target : PlotObject
        The backend object representing a :term:`plot` where this
        :term:`visual` should be added.
    size : any, optional
        Size of the label text.
    color : any, optional
        Color of the label text.
        If needed, see :ref:`backend_interface_arguments` for more details.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.set_xlabel`
        * plotly -> :class:`~plotly.graph_objects.layout.XAxis`
        * bokeh -> :class:`~bokeh.models.Axis`

    Returns
    -------
    xlabel_visual : any
        The backend object representing the generated x-axis label.
    """
    kwargs = {"color": color, "size": size}
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "xlabel", "string": string, **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def xticks(ticks, labels, target, *, rotation=unset, **artist_kws):
    """Interface to x-axis ticks.

    Parameters
    ----------
    ticks : array-like
        Positions of the ticks.
    labels : array-like
        Labels corresponding to the tick positions.
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    rotation : float or int, optional
        Rotation angle of the tick labels.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.set_xticks`
        * plotly -> :class:`~plotly.graph_objects.layout.XAxis`
        * bokeh -> :class:`~bokeh.models.Axis`
    Returns
    -------
    xticks_visual : any
        The backend object representing the x-axis ticks.
    """
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
    """Interface to y-axis ticks.

    Parameters
    ----------
    ticks : array-like
        Positions of the ticks.
    labels : array-like
        Labels corresponding to the tick positions.
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    rotation : float or int, optional
        Rotation angle of the tick labels.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.set_yticks`
        * plotly -> :class:`~plotly.graph_objects.layout.YAxis`
        * bokeh -> :class:`~bokeh.models.Axis`

    Returns
    -------
    yticks_visual : any
        The backend object representing the y-axis ticks.
    """
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
    """Interface to setting tick label visibility.

    Parameters
    ----------
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    axis : {"x", "y", "both"}, optional
        Axis for which tick label visibility should be modified. Defaults to "both".
    visible : bool, optional
        Whether the tick labels should be visible. Defaults to True.

    Returns
    -------
    set_ticklabel_visibility_visual : any
        The backend object representing the tick label visibility setting.
    """
    artist_element = {
        "function": "set_ticklabel_visibility",
        "axis": axis,
        "visible": visible,
    }
    target.append(artist_element)
    return artist_element


def xlim(lims, target, **artist_kws):
    """Interface to x-axis limits.

    Parameters
    ----------
    lims : tuple of float
        Lower and upper limits of the x-axis.
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.set_xlim`
        * plotly -> :class:`~plotly.graph_objects.layout.XAxis`
        * bokeh -> :class:`~bokeh.models.Range1d`
    Returns
    -------
    xlim_visual : any
        The backend object representing the x-axis limits.
    """
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "xlim", "lims": lims, **artist_kws}
    target.append(artist_element)
    return artist_element


def ylim(lims, target, **artist_kws):
    """Interface to y-axis limits.

    Parameters
    ----------
    lims : tuple of float
        Lower and upper limits of the y-axis.
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.set_ylim`
        * plotly -> :class:`~plotly.graph_objects.layout.YAxis`
        * bokeh -> :class:`~bokeh.models.Range1d`

    Returns
    -------
    ylim_visual : any
        The backend object representing the y-axis limits.
    """
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    artist_element = {"function": "ylim", "lims": lims, **artist_kws}
    target.append(artist_element)
    return artist_element


def ticklabel_props(target, *, axis="both", size=unset, color=unset, **artist_kws):
    """Interface to tick label properties.

    Parameters
    ----------
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    axis : {"x", "y", "both"}, optional
        Axis for which tick label properties should be modified. Defaults to "both".
    size : any, optional
        Size of the tick labels.
    color : any, optional
        Color of the tick labels.
    **artist_kws
        Passed to the backend plotting function of the respective backend:

        * matplotlib -> :meth:`~matplotlib.axes.Axes.tick_params`
        * plotly -> :class:`~plotly.graph_objects.layout.XAxis` /
        :class:`~plotly.graph_objects.layout.YAxis`
        * bokeh -> :class:`~bokeh.models.Axis`

    Returns
    -------
    ticklabel_props_visual : any
        The backend object representing the tick label properties.
    """
    if not ALLOW_KWARGS and artist_kws:
        raise ValueError(f"artist_kws not empty: {artist_kws}")
    kwargs = {"axis": axis, "size": size, "color": color}
    artist_element = {"function": "ticklabel_props", **_filter_kwargs(kwargs, artist_kws)}
    target.append(artist_element)
    return artist_element


def remove_ticks(target, *, axis="y"):
    """Interface to removing ticks from a plot.

    Parameters
    ----------
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    axis : {"x", "y"}, default "y"
        Axis from which the ticks should be removed.

    Returns
    -------
    remove_ticks_visual : any
        The backend object representing the removal of ticks.
    """
    artist_element = {
        "function": "remove_ticks",
        "axis": axis,
    }
    target.append(artist_element)
    return artist_element


def remove_axis(target, axis="y"):
    """Interface to removing an axis from a plot.

    Parameters
    ----------
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    axis : {"x", "y"}, default "y"
        Axis that should be removed.

    Returns
    -------
    remove_axis_visual : any
        The backend object representing the removal of the axis.
    """
    artist_element = {
        "function": "remove_axis",
        "axis": axis,
    }
    target.append(artist_element)
    return artist_element


def set_y_scale(target, scale):
    """Interface to setting the y-axis scale of a plot.

    Parameters
    ----------
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    scale : str
        Scale to apply to the y-axis (e.g. "linear", "log").

    Returns
    -------
    set_y_scale_visual : any
        The backend object representing the updated y-axis scale.
    """
    artist_element = {
        "function": "set_y_scale",
        "scale": scale,
    }
    target.append(artist_element)
    return artist_element


def grid(target, axis, color):
    """Interface to a grid.

    Parameters
    ----------
    target : PlotObject
        The backend object representing a plot where this visual should be added.
    axis : {"x", "y", "both"}
        Axis on which the grid should be applied.
    color : any
        Color of the grid lines.

    Returns
    -------
    grid_visual : any
        The backend object representing the grid.
    """
    artist_element = {
        "function": "grid",
        "axis": axis,
        "color": color,
    }
    target.append(artist_element)
    return artist_element
