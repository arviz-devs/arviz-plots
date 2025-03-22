"""Bokeh interface layer."""
# pylint: disable=protected-access

import math
import warnings

import numpy as np
from bokeh.layouts import GridBox, gridplot
from bokeh.models import (
    ColumnDataSource,
    CustomJSTickFormatter,
    FixedTicker,
    GridPlot,
    Range1d,
    Span,
    Title,
)
from bokeh.plotting import figure
from bokeh.plotting import show as _show

from ..none import get_default_aes as get_agnostic_default_aes
from .legend import legend


class UnsetDefault:
    """Specific class to indicate an aesthetic hasn't been set."""


unset = UnsetDefault()


def set_sqrt_yscale(target):
    """Transform existing plots on a figure to use sqrt(y) scale."""
    max_y = 0
    for renderer in target.renderers:
        if isinstance(renderer.data_source, ColumnDataSource):
            ds = renderer.data_source
            if "y" in ds.data:
                current_max = max(ds.data["y"])
                max_y = max(max_y, current_max)
            elif "y_top" in ds.data:
                current_max = max(ds.data["y_top"])
                max_y = max(max_y, current_max)

    ticks = []
    if max_y > 0:
        for i in range(math.ceil(np.sqrt(max_y)) + 1):
            ticks.append(i)

    target.yaxis.formatter = CustomJSTickFormatter(
        code="""
        return (tick ** 2).toFixed(0)
    """
    )
    target.yaxis.ticker = FixedTicker(ticks=ticks)
    target.y_range.start = 0
    target.y_range.end = ticks[len(ticks) - 1]

    # Transform existing scatter plots
    for renderer in target.renderers:
        if hasattr(renderer.glyph, "y") and isinstance(renderer.data_source, ColumnDataSource):
            ds = renderer.data_source
            y_field = renderer.glyph.y

            if "original_y" in ds.data:
                continue

            original_y = ds.data[y_field]
            ds.data["original_y"] = original_y
            ds.data["y_sqrt"] = np.sqrt(original_y)

            renderer.glyph.y = "y_sqrt"
        elif (
            hasattr(renderer.glyph, "y0")
            and hasattr(renderer.glyph, "y1")
            and isinstance(renderer.data_source, ColumnDataSource)
        ):
            ds = renderer.data_source
            y_top_field = renderer.glyph.y1
            y_bottom_field = renderer.glyph.y0

            if "original_y_top" in ds.data and "original_y_bottom" in ds.data:
                continue

            original_y_top = ds.data[y_top_field]
            ds.data["original_y_top"] = original_y_top
            ds.data["y_top_sqrt"] = np.sqrt(original_y_top)

            original_y_bottom = ds.data[y_bottom_field]
            ds.data["original_y_bottom"] = original_y_bottom
            ds.data["y_bottom_sqrt"] = np.sqrt(original_y_bottom)

            renderer.glyph.y0 = "y_bottom_sqrt"
            renderer.glyph.y1 = "y_top_sqrt"


# generation of default values for aesthetics
def get_default_aes(aes_key, n, kwargs=None):
    """Generate `n` *bokeh valid* default values for a given aesthetics keyword."""
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
            try:
                from bokeh.io import curdoc

                template_colors = curdoc().theme._json["attrs"]["Cycler"]["colors"]
            except (ImportError, KeyError):
                template_colors = None
            vals = vals if template_colors is None else template_colors

        elif aes_key in {"linestyle", "line_dash"}:
            vals = ["solid", "dashed", "dotted", "dashdot"]
        elif aes_key == "marker":
            vals = ["circle", "cross", "triangle", "x", "diamond", "square", "dot"]
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
def show(chart):
    """Show the provided bokeh layout."""
    _show(chart)


def get_figsize(plot_collection):
    """Get the size of the :term:`chart` element and its units."""
    chart = plot_collection.viz["chart"].item()
    if chart is None:
        plot = plot_collection.viz["plot"].item()
        return (plot.width, plot.height), "dots"
    if isinstance(chart, (GridBox, GridPlot)):
        gridbox = chart
    elif isinstance(chart, tuple):
        gridbox = chart[1]
    else:
        gridbox = chart.children[1]
    if not isinstance(gridbox, (GridBox, GridPlot)):
        return (800, 800)
    row_heights_sum = np.sum([plot.height for plot, _, col in gridbox.children if col == 0])
    col_widths_sum = np.sum([plot.width for plot, row, _ in gridbox.children if row == 0])
    return (col_widths_sum, row_heights_sum), "dots"


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
        Size of the figure in `figsize_units`. It overwrites the values for "width" and "height"
        in `subplot_kws` if any.
    figsize_units : {"inches", "dots"}, default "inches"
        Units in which `figsize` is given.
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

    if plot_hspace is not None:
        subplot_kws.setdefault("min_border_left", plot_hspace)
        subplot_kws.setdefault("min_border_right", plot_hspace)

    if figsize is not None:
        if figsize_units == "inches":
            figsize = (figsize[0] * 100, figsize[1] * 100)
            warnings.warn(
                f"Assuming dpi=100. Use figsize_units='dots' and figsize={figsize} "
                "to stop seeing this warning"
            )
        elif figsize_units != "dots":
            raise ValueError(f"figsize_units must be 'dots' or 'inches', but got {figsize_units}")
        subplot_kws["width"] = int(np.ceil(figsize[0] / cols))
        subplot_kws["height"] = int(np.ceil(figsize[1] / rows))

    plot_widths = None
    if width_ratios is not None:
        if len(width_ratios) != cols:
            raise ValueError("width_ratios must be an iterable of length cols")
        plot_width = subplot_kws.get("width", 600)
        chart_width = plot_width * cols
        width_ratios = np.array(width_ratios, dtype=float)
        width_ratios /= width_ratios.sum()
        plot_widths = np.ceil(chart_width * width_ratios).astype(int)

    shared_xrange = {}
    shared_yrange = {}
    for row in range(rows):
        for col in range(cols):
            subplot_kws_i = subplot_kws.copy()
            if col != 0 and sharex == "row":
                subplot_kws_i["x_range"] = shared_xrange[row]
            if row != 0 and sharex == "col":
                subplot_kws_i["x_range"] = shared_xrange[col]
            if col != 0 and sharey == "row":
                subplot_kws_i["y_range"] = shared_yrange[row]
            if row != 0 and sharey == "col":
                subplot_kws_i["y_range"] = shared_yrange[col]
            if width_ratios is not None:
                subplot_kws["width"] = plot_widths[col]
            if (row == 0) and (col == 0) and (sharex is True or sharey is True):
                p = figure(**subplot_kws_i)  # pylint: disable=invalid-name
                figures[row, col] = p
                if sharex is True:
                    subplot_kws["x_range"] = p.x_range
                if sharey is True:
                    subplot_kws["y_range"] = p.y_range
            elif col == 0 and (sharex == "row" or sharey == "row"):
                p = figure(**subplot_kws_i)  # pylint: disable=invalid-name
                figures[row, col] = p
                if sharex == "row":
                    shared_xrange[row] = p.x_range
                if sharey == "row":
                    shared_yrange[row] = p.y_range
            elif row == 0 and (sharex == "col" or sharey == "col"):
                p = figure(**subplot_kws_i)  # pylint: disable=invalid-name
                figures[row, col] = p
                if sharex == "col":
                    shared_xrange[col] = p.x_range
                if sharey == "col":
                    shared_yrange[col] = p.y_range
            elif row * cols + (col + 1) > number:
                figures[row, col] = None
            else:
                figures[row, col] = figure(**subplot_kws_i)
    if squeeze and figures.size == 1:
        return None, figures[0, 0]
    layout = gridplot(figures.tolist(), **kwargs)
    return layout, figures.squeeze() if squeeze else figures


# helper functions
def _filter_kwargs(kwargs, artist_kws):
    """Filter a dictionary to remove all keys whose values are ``unset``."""
    kwargs = {key: value for key, value in kwargs.items() if value is not unset}
    return {**artist_kws, **kwargs}


def _float_or_str_size(size):
    """Bokeh only accepts string sizes with units.

    Convert float sizes to string ones in px units.
    """
    if size is unset:
        return size
    if isinstance(size, str):
        return size
    return f"{size:.0f}px"


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
    step_mode="center",
    **artist_kws,
):
    """Interface to Bokeh for a histogram bar or step plot."""
    if color is not unset:
        if facecolor is unset:
            facecolor = color
        if edgecolor is unset:
            edgecolor = color

    step_hist = artist_kws.pop("step", False)
    kwargs = {"bottom": bottom, "fill_color": facecolor, "line_color": edgecolor, "alpha": alpha}
    if step_hist:
        kwargs = {"line_color": edgecolor, "alpha": alpha}

        x = [l_e[0], l_e[0]]
        y_step = [0, y[0]]
        for i, y_i in enumerate(y):
            x.append(r_e[i])
            y_step.append(y_i)
        x.append(r_e[-1])
        y_step.append(bottom)

        p = target.step(x, y_step, mode=step_mode, **_filter_kwargs(kwargs, artist_kws))

        target.x_range = Range1d(float(l_e[0]), float(r_e[-1]))
        target.y_range = Range1d(float(bottom), float(max(y)) * 1.2)  # Add padding to y-axis

        return p

    return target.quad(top=y, left=l_e, right=r_e, **_filter_kwargs(kwargs, artist_kws))


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

    source = ColumnDataSource(data={"x": np.atleast_1d(x), "y": np.atleast_1d(y)})
    return target.scatter(x="x", y="y", source=source, **kwargs)


def step(x, y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to bokeh for a step line."""
    kwargs = {"color": color, "alpha": alpha, "line_width": width, "line_dash": linestyle}
    return target.step(np.atleast_1d(x), np.atleast_1d(y), **_filter_kwargs(kwargs, artist_kws))


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
        "text_font_size": _float_or_str_size(size),
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
    y_bottom = np.atleast_1d(y_bottom)
    if y_bottom.size == 1:
        y_bottom = y_bottom.item()
    y_top = np.atleast_1d(y_top)
    if y_top.size == 1:
        y_top = y_top.item()
    return target.varea(x=x, y1=y_bottom, y2=y_top, **artist_kws)


def vline(x, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to bokeh for a vertical line spanning the whole axes."""
    kwargs = {"line_color": color, "line_alpha": alpha, "line_width": width, "line_dash": linestyle}
    span_element = Span(location=x, dimension="height", **_filter_kwargs(kwargs, artist_kws))
    target.add_layout(span_element)
    return span_element


def hline(y, target, *, color=unset, alpha=unset, width=unset, linestyle=unset, **artist_kws):
    """Interface to bokeh for a horizontal line spanning the whole axes."""
    kwargs = {"line_color": color, "line_alpha": alpha, "line_width": width, "line_dash": linestyle}
    span_element = Span(location=y, dimension="width", **_filter_kwargs(kwargs, artist_kws))
    target.add_layout(span_element)
    return span_element


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
    """Interface to bokeh for a line from y_bottom to y_top at given value of x."""
    kwargs = {"color": color, "alpha": alpha, "line_width": width, "line_dash": linestyle}
    x = np.atleast_1d(x)
    y_bottom = np.atleast_1d(y_bottom)
    y_top = np.atleast_1d(y_top)
    source = ColumnDataSource(
        data={
            "x": np.atleast_1d(x),
            "y_bottom": np.atleast_1d(y_bottom),
            "y_top": np.atleast_1d(y_top),
        }
    )
    return target.segment(
        x0="x",
        x1="x",
        y0="y_bottom",
        y1="y_top",
        source=source,
        **_filter_kwargs(kwargs, artist_kws),
    )


# general plot appeareance
def title(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to bokeh for adding a title to a plot."""
    kwargs = {"text_font_size": _float_or_str_size(size), "text_color": color}
    target.title = Title(text=string, **_filter_kwargs(kwargs, artist_kws))
    return target.title


def ylabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to bokeh for adding a label to the y axis."""
    kwargs = {"text_font_size": _float_or_str_size(size), "text_color": color}
    target.yaxis.axis_label = string
    for key, value in _filter_kwargs(kwargs, artist_kws).items():
        setattr(target.yaxis, f"axis_label_{key}", value)


def xlabel(string, target, *, size=unset, color=unset, **artist_kws):
    """Interface to bokeh for adding a label to the x axis."""
    kwargs = {"text_font_size": _float_or_str_size(size), "text_color": color}
    target.xaxis.axis_label = string
    for key, value in _filter_kwargs(kwargs, artist_kws).items():
        setattr(target.xaxis, f"axis_label_{key}", value)


def xticks(ticks, labels, target, **artist_kws):
    """Interface to bokeh for setting ticks and labels of the x axis."""
    target.xaxis.ticker = ticks
    if labels is not None:
        target.xaxis.major_label_overrides = {
            key.item() if hasattr(key, "item") else key: value for key, value in zip(ticks, labels)
        }
    for key, value in _filter_kwargs({}, artist_kws).items():
        setattr(target.xaxis, f"major_label_{key}", value)


def yticks(ticks, labels, target, **artist_kws):
    """Interface to bokeh for setting ticks and labels of the y axis."""
    target.yaxis.ticker = ticks
    if labels is not None:
        target.yaxis.major_label_overrides = {
            key.item() if hasattr(key, "item") else key: value for key, value in zip(ticks, labels)
        }
    for key, value in _filter_kwargs({}, artist_kws).items():
        setattr(target.yaxis, f"major_label_{key}", value)


def xlim(lims, target, **artist_kws):
    """Interface to bokeh for setting limits for the x axis."""
    target.x_range = Range1d(*lims, **artist_kws)


def ticklabel_props(target, *, axis="both", size=unset, color=unset, **artist_kws):
    """Interface to bokeh for setting ticks size."""
    kwargs = {"text_font_size": _float_or_str_size(size), "text_color": color}
    for key, value in _filter_kwargs(kwargs, artist_kws).items():
        if axis in {"y", "both"}:
            setattr(target.yaxis, f"major_label_{key}", value)
        if axis in {"x", "both"}:
            setattr(target.xaxis, f"major_label_{key}", value)


def remove_ticks(target, *, axis="y"):  # pylint: disable=unused-argument
    """Interface to bokeh for removing ticks from a plot."""
    if axis in {"y", "both"}:
        target.yaxis.major_tick_out = 0
        target.yaxis.major_tick_in = 0
        target.yaxis.major_label_text_font_size = "0pt"
    if axis in {"x", "both"}:
        target.xaxis.major_tick_out = 0
        target.xaxis.major_tick_in = 0
        target.xaxis.major_label_text_font_size = "0pt"


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


def set_y_scale(target, scale):
    """Interface to bokeh for setting the y scale of a plot."""
    if scale == "sqrt":
        set_sqrt_yscale(target)
    else:
        pass


def grid(target, axis, color):
    """Interface to bokeh for setting a grid in any axis."""
    if axis in ["y", "both"]:
        target.ygrid.grid_line_color = color
    if axis in ["x", "both"]:
        target.xgrid.grid_line_color = color
