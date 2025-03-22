"""Plotly interface layer.

Notes
-----
:term:`artists` are returned, but it seems modifying them won't modify the figure.
"""

import math
import re
import warnings

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from webcolors import hex_to_rgb, name_to_rgb

from ..none import get_default_aes as get_agnostic_default_aes


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()
pat = re.compile(r"^(row|col)\s?:")


def apply_square_root_scale(plotly_plot):
    """Apply a square root scale to the y-axis of a PlotlyPlot."""
    chart = plotly_plot.chart
    row = plotly_plot.row
    col = plotly_plot.col

    if not hasattr(chart.layout, "grid") or chart.layout.grid is None:
        raise ValueError("The chart does not have a grid layout required for faceting.")

    chart_grid = chart.layout.grid
    num_cols = chart_grid.columns if chart_grid.columns is not None else 1
    index = (row - 1) * num_cols + col

    yaxis_ref = "y" if index == 1 else f"y{index}"
    layout_yaxis = "yaxis" if index == 1 else f"yaxis{index}"

    y_transformed_all = []
    for trace in chart.data:
        if getattr(trace, "yaxis", None) == yaxis_ref:
            if hasattr(trace, "y") and trace.y is not None:
                y_data = np.array(trace.y, dtype=float)
                y_data = np.maximum(y_data, 0.0)  # Clamp negative values
                y_transformed = np.sqrt(y_data)
                trace.y = y_transformed.tolist()
                y_transformed_all.extend(y_transformed)

    if not y_transformed_all:
        return

    y_min = min(y_transformed_all)
    y_max = max(y_transformed_all)

    tickvals_transformed = []
    for i in range(int(math.ceil(y_max)) + 1):
        tickvals_transformed.append(i)

    if len(tickvals_transformed) == 0:
        tickvals_transformed = np.array([y_min, y_max])

    ticktext_original = [f"{round(tv**2)}" for tv in tickvals_transformed]

    chart.layout[layout_yaxis].update(
        tickvals=tickvals_transformed,
        ticktext=ticktext_original,
        title=chart.layout[layout_yaxis].title,
    )

    chart.layout[layout_yaxis].range = [y_min - 0.1, y_max + 0.1]


def str_to_plotly_html(string):
    """Convert input string to html subset used by plotly."""
    return string.replace("\n", "<br>")


def combine_color_alpha(color, alpha=1):
    """Combine a color and alpha value into the equivalent rgba."""
    if isinstance(color, str):
        if color.startswith("rgba("):
            warnings.warn("Found rgba color, value for `alpha` is ignored.")
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
def get_default_aes(aes_key, n, kwargs=None):
    """Generate `n` *plotly valid* default values for a given aesthetics keyword."""
    if kwargs is None:
        kwargs = {}
    if aes_key not in kwargs:
        if "color" in aes_key:
            # fmt: off
            vals = [
                '#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6',
                '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd'
            ]
            # fmt: on
            template_colors = pio.templates[pio.templates.default].layout.colorway
            vals = vals if template_colors is None else template_colors
        elif aes_key in {"linestyle", "dash"}:
            vals = ["solid", "dash", "dot", "dashdot"]
        elif aes_key in {"marker", "style"}:
            # plotly does not have "dot" using "circle-open" instead
            vals = ["circle", "cross", "triangle-up", "x", "diamond", "square", "circle-open"]
        else:
            return get_agnostic_default_aes(aes_key, n, {})
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

    Plotly supports :term:`faceting` but it doesn't have any object that represents
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
    *,
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
    share_lookup = {True: "all", "col": "columns", "row": "rows"}
    sharex = share_lookup.get(sharex, sharex)
    sharey = share_lookup.get(sharey, sharey)

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
        return chart, plots[0, 0]
    return chart, plots.squeeze() if squeeze else plots


# helper functions
def _filter_kwargs(kwargs, artist_kws):
    """Filter a dictionary to remove all keys whose values are ``unset``."""
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
    alpha=unset,
    facecolor=unset,
    edgecolor=unset,
    **artist_kws,
):
    """Interface to Plotly for a histogram plot."""
    artist_kws.setdefault("showlegend", False)
    step_hist = artist_kws.pop("step", False)
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

    if step_hist:
        step_x = np.repeat(np.concatenate(([l_e[0]], r_e)), 2)[1:-1]
        step_y = np.repeat(np.concatenate(([0], height)), 2)
        hist_object = go.Scatter(
            x=step_x,
            y=step_y,
            fill="none",
            line={"color": edgecolor},
            mode="lines",
            **_filter_kwargs({"opacity": alpha}, artist_kws),
        )
    else:
        marker_artist_kws = artist_kws.pop("marker", {}).copy()
        line_kwargs = {"color": edgecolor}
        line_artist_kws = marker_artist_kws.pop("line", {}).copy()
        marker_kwargs = _filter_kwargs({"color": facecolor}, marker_artist_kws)
        marker_kwargs["line"] = _filter_kwargs(line_kwargs, line_artist_kws)
        hist_object = go.Bar(
            x=(l_e + r_e) / 2,
            y=height,
            base=bottom,
            width=widths,
            marker=marker_kwargs,
            **_filter_kwargs({"opacity": alpha}, artist_kws),
        )

    target.add_trace(hist_object)
    return hist_object


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


def step(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to plotly for a step line."""
    artist_kws.setdefault("showlegend", False)
    line_kwargs = {"color": color, "width": width, "dash": linestyle}
    line_artist_kws = artist_kws.pop("line", {"shape": "hv"}).copy()
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


def vline(x, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to plotly for a vertical line spanning the whole axes."""
    artist_kws.setdefault("showlegend", False)
    line_kwargs = {"color": color, "width": width, "dash": linestyle}
    line_artist_kws = artist_kws.pop("line", {}).copy()
    kwargs = {"opacity": alpha}
    return target.add_vline(
        x, line=_filter_kwargs(line_kwargs, line_artist_kws), **_filter_kwargs(kwargs, artist_kws)
    )


def hline(y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to plotly for a horizontal line spanning the whole axes."""
    artist_kws.setdefault("showlegend", False)
    line_kwargs = {"color": color, "width": width, "dash": linestyle}
    line_artist_kws = artist_kws.pop("line", {}).copy()
    kwargs = {"opacity": alpha}
    return target.add_hline(
        y, line=_filter_kwargs(line_kwargs, line_artist_kws), **_filter_kwargs(kwargs, artist_kws)
    )


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
    """Interface to plotly for a line from y_bottom to y_top at given value of x."""
    artist_kws.setdefault("showlegend", False)
    line_kwargs = {"color": color, "width": width, "dash": linestyle}
    line_artist_kws = artist_kws.pop("line", {}).copy()
    kwargs = {"opacity": alpha}

    # I was not able to figure it out how to do this withouth a loop
    # all solutions I tried did not plot the lines or the lines were connected
    for x_i, y_t, y_b in zip(x, y_top, y_bottom):
        line_object = go.Scatter(
            x=[x_i, x_i],
            y=[y_b, y_t],
            mode="lines",
            line=_filter_kwargs(line_kwargs, line_artist_kws),
            **_filter_kwargs(kwargs, artist_kws),
        )
        target.add_trace(line_object)
    return line_object


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
        title={"text": str_to_plotly_html(string), "font": _filter_kwargs(kwargs, artist_kws)}
    )


def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to plotly for adding a label to the y axis."""
    kwargs = {"size": size, "color": color}
    target.update_xaxes(
        title={"text": str_to_plotly_html(string), "font": _filter_kwargs(kwargs, artist_kws)}
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


def set_y_scale(target, scale):
    """Interface to bokeh for setting the y scale of a plot."""
    if scale == "sqrt":
        apply_square_root_scale(target)
    else:
        pass


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


def grid(target, axis="both", color=unset, **artist_kws):
    """Interface to plotly for setting a grid in any axis."""
    kwargs = {"showgrid": True, "gridcolor": color}
    if axis in ["y", "both"]:
        target.update_yaxes(_filter_kwargs(kwargs, artist_kws))
    if axis in ["x", "both"]:
        target.update_xaxes(_filter_kwargs(kwargs, artist_kws))
