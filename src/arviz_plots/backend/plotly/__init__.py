"""Plotly interface layer.

Notes
-----
:term:`artists` are returned, but it seems modifying them won't modify the figure.
"""
import re
import warnings

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from webcolors import hex_to_rgb, name_to_rgb

from .. import get_default_aes as get_agnostic_default_aes


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()
pat = re.compile(r"^(row|col)\s?:")


def str_to_plotly_html(string):
    """Convert input string to html subset used by plotly."""
    return string.replace("\n", "<br>")


def combine_color_alpha(color, alpha=1):
    """Combine a color and alpha value into the equivalent rgba."""
    if isinstance(color, str):
        if color.startswith("rgba("):
            warnings.warning("Found rgba color, value for `alpha` is ignored.")
            return color
        if color.startswith("#"):
            color = hex_to_rgb(color)
        elif color.startswith("rgb("):
            color = color.strip("rgb()").split(",")
        else:
            color = name_to_rgb(color)
    if len(color) != 3:
        raise ValueError("Invalid color")
    return f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha:.3f})"


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs):
    """Generate `n` *plotly valid* default values for a given aesthetics keyword."""
    if aes_key not in kwargs:
        if "color" in aes_key:
            # fmt: off
            vals = [
                '#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6',
                '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd'
            ]
            # fmt: on
        elif aes_key in {"linestyle", "dash"}:
            vals = ["solid", "dash", "dot", "dashdot"]
        elif aes_key in {"marker", "style"}:
            vals = ["circle", "cross", "triangle-up", "x", "diamond"]
        else:
            return get_agnostic_default_aes(aes_key, n, {})
        return get_agnostic_default_aes(aes_key, n, {aes_key: vals})
    return get_agnostic_default_aes(aes_key, n, kwargs)


def scale_fig_size(figsize, rows=1, cols=1, figsize_units="inches"):
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
    if figsize is None:
        width = 800
        height = (100 * rows + 100) ** 1.1
        figsize_units = "dots"
    else:
        width, height = figsize
    cols = cols * 100
    rows = rows * 100
    if figsize_units == "inches":
        warnings.warn(
            f"Assuming dpi=100. Use figsize_units='dots' and figsize={figsize} "
            "to stop seeing this warning"
        )
        width *= 100
        height *= 100
    elif figsize_units != "dots":
        raise ValueError(f"figsize_units must be 'dots' or 'inches', but got {figsize_units}")

    val = (width * height) ** 0.5
    val2 = (cols * rows) ** 0.5
    scale_factor = val / (4 * val2)
    # I didn't find any Plotly equivalent to theme/rcParams,
    # so they are hardcoded for now
    labelsize = 14 * scale_factor
    linewidth = 1 * scale_factor

    return (width, height), labelsize, linewidth


def get_figsize(plot_collection):
    """Get the size of the :term:`chart` element and its units."""
    chart = plot_collection.viz["chart"].item()
    if chart is None:
        chart = plot_collection.viz["plot"].item()
    height = chart.layout.height
    width = chart.layout.width
    return (800 if width is None else width, 800 if height is None else height), "dots"


def remove_row_col_from_doc(docstring):
    """Remove the row and col parameters and their description from a docstring."""
    doc_lines = docstring.split("\n")
    for doc_line in doc_lines:
        if doc_line:
            first_line = doc_line
            break
    idx = np.argmin([c == " " for c in first_line])
    doc_lines = [line[idx:] for line in doc_lines]
    in_row_col = False
    new_lines = []
    for doc_line in doc_lines:
        if in_row_col:
            if doc_line.startswith(" "):
                continue
            in_row_col = False
        if pat.match(doc_line):
            in_row_col = True
            continue
        new_lines.append(doc_line)
    return "\n".join(new_lines)


class PlotlyPlot:
    """Custom class to represent a :term:`plot` in Plotly.

    Plotly supports :term:`facetting` but it doesn't have any object that represents
    a :term:`plot`, instead, plotting happens only though the Plotly figure
    (which represents the :term:`chart`) indicating the row and column indexes
    in case plotting should happen to a single :term:`plot`.

    This class is initialized with the plotly figure and the row and col indexes,
    then it exposes all methods of the plotly figure object with the row and col
    arguments already set.
    """

    def __init__(self, chart, row, col):
        self.row = row
        self.col = col
        self.chart = chart

    def __getattr__(self, name):
        """Expose all methods of the plotly figure with row and col arguments set."""
        if hasattr(self.chart, name):
            original_fun = getattr(self.chart, name)

            def aux_fun(*args, **kwargs):
                return original_fun(*args, **kwargs, row=self.row, col=self.col)

            aux_fun.__doc__ = remove_row_col_from_doc(original_fun.__doc__)
            return aux_fun
        raise AttributeError(f"Attribute {name} not available")


# object creation and i/o
def show(chart):
    """Show the provided plotly layout."""
    chart.show()


def create_plotting_grid(
    number,  # pylint: disable=unused-argument
    rows=1,
    cols=1,
    figsize=None,
    figsize_units="inches",
    squeeze=True,
    sharex=False,
    sharey=False,
    polar=False,  # pylint: disable=unused-argument
    width_ratios=None,
    plot_hspace=None,
    subplot_kws=None,  # pylint: disable=unused-argument
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
        Size of the figure in `figsize_units`. It overwrites the values for "width" and "height"
        in `subplot_kws` if any.
    figsize_units : {"inches", "dots"}, default "inches"
        Units in which `figsize` is given.
    squeeze : bool, default True
    sharex, sharey : bool, default False
    polar : bool
    subplot_kws : bool
        Ignored
    **kwargs: dict, optional
        Passed to :func:`~plotly.subplots.make_subplots`

    Returns
    -------
    `~plotly.graph_object.Figure` or None
    `~arviz_plots.backend.plotly.PlotlyPlot` or ndarray of `~arviz_plots.backend.plotly.PlotlyPlot`
    """
    plots = np.empty((rows, cols), dtype=object)
    layout_kwargs = {}  # {"legend": {"visible": False}}

    if figsize is not None:
        if figsize_units == "inches":
            figsize = (figsize[0] * 100, figsize[1] * 100)
            warnings.warn(
                f"Assuming dpi=100. Use figsize_units='dots' and figsize={figsize} "
                "to stop seeing this warning"
            )
        elif figsize_units != "dots":
            raise ValueError(f"figsize_units must be 'dots' or 'inches', but got {figsize_units}")
        layout_kwargs["width"] = figsize[0]
        layout_kwargs["height"] = figsize[1]

    kwargs["figure"] = go.Figure(layout=layout_kwargs)

    chart = make_subplots(
        rows=int(rows),
        cols=int(cols),
        shared_xaxes=sharex,
        shared_yaxes=sharey,
        start_cell="top-left",
        horizontal_spacing=plot_hspace,
        column_widths=width_ratios if width_ratios is None else list(width_ratios),
        **kwargs,
    )

    for row in range(rows):
        for col in range(cols):
            plots[row, col] = PlotlyPlot(chart, row + 1, col + 1)
    if squeeze and plots.size == 1:
        return None, plots[0, 0]
    return chart, plots.squeeze() if squeeze else plots


# helper functions
def _filter_kwargs(kwargs, artist_kws):
    """Filter a dictionary to remove all keys whose values are ``unset``."""
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    return {**artist_kws, **kwargs}


# "geoms"
def line(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to plotly for a line plot."""
    artist_kws.setdefault("showlegend", False)
    line_kwargs = {"color": color, "width": width, "dash": linestyle}
    line_artist_kws = artist_kws.pop("line", {}).copy()
    kwargs = {"opacity": alpha}
    line_object = go.Scatter(
        x=np.atleast_1d(x),
        y=np.atleast_1d(y),
        mode="lines",
        line=_filter_kwargs(line_kwargs, line_artist_kws),
        **_filter_kwargs(kwargs, artist_kws),
    )
    target.add_trace(line_object)
    return line_object


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
    """Interface to plotly for a scatter plot."""
    artist_kws.setdefault("showlegend", False)
    if color is not unset:
        if facecolor is unset and edgecolor is unset:
            facecolor = color
            edgecolor = color
        elif facecolor is unset:
            facecolor = color
        elif edgecolor is unset:
            edgecolor = color
    marker_artist_kws = artist_kws.pop("marker", {}).copy()
    edgeline_artist_kws = marker_artist_kws.pop("line", {}).copy()
    line_kwargs = _filter_kwargs({"color": edgecolor, "width": width}, edgeline_artist_kws)
    scatter_kwargs = {
        "size": size if size is unset else np.sqrt(size),
        "symbol": marker,
        "opacity": alpha,
        "color": facecolor,
        "line": line_kwargs if line_kwargs else unset,
    }
    if marker == "|":
        scatter_kwargs["symbol"] = "line-ns-open"
    scatter_object = go.Scatter(
        x=np.atleast_1d(x),
        y=np.atleast_1d(y),
        mode="markers",
        marker=_filter_kwargs(scatter_kwargs, marker_artist_kws),
        **artist_kws,
    )
    target.add_trace(scatter_object)
    return scatter_object


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
    """Interface to plotly for adding text to a plot."""
    artist_kws.setdefault("showlegend", False)
    # plotly inverts the meaning of alignment with respect to matplotlib and bokeh
    vertical_align = {"top": "bottom", "bottom": "top"}.get(vertical_align, vertical_align)
    horizontal_align = {"right": "left", "left": "right"}.get(horizontal_align, horizontal_align)

    textfont_artist_kws = artist_kws.pop("textfont", {}).copy()
    text_kwargs = {"color": color, "size": size}
    kwargs = {"opacity": alpha}
    text_object = go.Scatter(
        x=np.atleast_1d(x),
        y=np.atleast_1d(y),
        text=np.vectorize(str_to_plotly_html)(np.atleast_1d(string)),
        mode="text",
        textfont=_filter_kwargs(text_kwargs, textfont_artist_kws),
        textposition=f"{vertical_align} {horizontal_align}",
        **_filter_kwargs(kwargs, artist_kws),
    )
    target.add_trace(text_object)
    return text_object


def fill_between_y(x, y_bottom, y_top, target, *, color=unset, alpha=unset, **artist_kws):
    """Interface to plotly for plotting a filled area between two curves."""
    artist_kws.setdefault("showlegend", False)
    kwargs = {"fillcolor": combine_color_alpha(color, alpha)}
    first_line = go.Scatter(
        x=np.atleast_1d(x),
        y=np.atleast_1d(y_bottom),
        mode="lines",
        line={"width": 0},
        fill=None,
        showlegend=False,
    )
    target.add_trace(first_line)
    second_line_with_fill = go.Scatter(
        x=np.atleast_1d(x),
        y=np.atleast_1d(y_top),
        fill="tonexty",
        mode="none",
        **_filter_kwargs(kwargs, artist_kws),
    )
    target.add_trace(second_line_with_fill)
    return second_line_with_fill


# general plot appeareance
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to plotly for adding a title to a plot."""
    kwargs = {"size": size, "color": color}
    font_kws = artist_kws.pop("font", {}).copy()
    title_object = go.layout.Annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=1.01,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        text=str_to_plotly_html(string),
        font=_filter_kwargs(kwargs, font_kws),
        **artist_kws,
    )
    target.add_annotation(title_object)
    return title_object


def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to plotly for adding a label to the y axis."""
    kwargs = {"size": size, "color": color}
    target.update_yaxes(
        title=str_to_plotly_html(string), titlefont=_filter_kwargs(kwargs, artist_kws)
    )


def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to plotly for adding a label to the y axis."""
    kwargs = {"size": size, "color": color}
    target.update_xaxes(
        title=str_to_plotly_html(string), titlefont=_filter_kwargs(kwargs, artist_kws)
    )


def xticks(ticks, labels, target, **artist_kws):
    """Interface to plotly for setting ticks and labels of the x axis."""
    if labels is None:
        labels = [str(label) for label in labels]
    target.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels, **artist_kws)


def yticks(ticks, labels, target, **artist_kws):
    """Interface to plotly for setting ticks and labels of the y axis."""
    if labels is None:
        labels = [str(label) for label in labels]
    target.update_yaxes(tickmode="array", tickvals=ticks, ticktext=labels, **artist_kws)


def xlim(lims, target, **artist_kws):
    """Interface to plotly for setting limits for the x axis."""
    target.update_xaxes(range=lims, **artist_kws)  # pylint: disable=redefined-builtin


def ticklabel_props(target, *, axis="both", size=unset, color=unset, **artist_kws):
    """Interface to plotly for setting ticks size."""
    kwargs = {"size": size, "color": color}
    if axis not in ("y", "x", "both"):
        raise ValueError(f"axis must be one of 'x', 'y' or 'both', got '{axis}'")
    if axis in {"y", "both"}:
        target.update_yaxes(tickfont=_filter_kwargs(kwargs, artist_kws))
    if axis in {"x", "both"}:
        target.update_xaxes(tickfont=_filter_kwargs(kwargs, artist_kws))


def remove_ticks(target, *, axis="y"):  # pylint: disable=unused-argument
    """Interface to plotly for removing ticks from a plot."""
    if axis not in ("y", "x", "both"):
        raise ValueError(f"axis must be one of 'x', 'y' or 'both', got '{axis}'")
    if axis in {"y", "both"}:
        target.update_yaxes(ticks="")
    if axis in {"x", "both"}:
        target.update_xaxes(ticks="")


def remove_axis(target, axis="y"):
    """Interface to plotly for removing axis from a plot."""
    if axis not in ("y", "x", "both"):
        raise ValueError(f"axis must be one of 'x', 'y' or 'both', got '{axis}'")
    if axis in ("y", "both"):
        target.update_yaxes(visible=False)
    if axis in ("x", "both"):
        target.update_xaxes(visible=False)
