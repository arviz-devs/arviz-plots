# File generated with docstub

import math
import warnings
from pathlib import Path
from typing import Any, Literal

import bokeh
import bokeh.colors.named as named_colors
import numpy as np
from _typeshed import Incomplete
from bokeh.colors import Color
from bokeh.io.export import export_png, export_svg
from bokeh.layouts import GridBox, column, gridplot
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    CustomJSTickFormatter,
    Div,
    FixedTicker,
    GridPlot,
    Range1d,
    Span,
    Title,
)
from bokeh.models.css import Styles
from bokeh.plotting import figure as _figure
from bokeh.plotting import output_file, save
from bokeh.plotting import show as _show
from numpy.typing import ArrayLike, NDArray

from arviz_plots.backend.alias_utils import create_aesthetic_handlers
from arviz_plots.backend.none import get_default_aes as get_agnostic_default_aes

class UnsetDefault:
    pass

unset: Incomplete

def _set_sqrt_scale(target: Incomplete, axis: Incomplete) -> None: ...
def get_hex_from_color_name(color_name: str) -> str: ...
def get_background_color() -> None: ...
def get_default_aes(
    aes_key: Incomplete, n: Incomplete, kwargs: Incomplete = ...
) -> None: ...

expand_aesthetic_aliases: Incomplete

def scale_fig_size(
    figsize: tuple[float, float] | None,
    rows: int = ...,
    cols: int = ...,
    figsize_units: Literal["inches", "dots"] | None = ...,
) -> tuple[tuple[float, float] | None, float, float]: ...
def show(figure: Incomplete) -> None: ...
def savefig(figure: bokeh.models.Plot, path: Path, **kwargs: Incomplete) -> None: ...
def set_figure_title(
    figure: bokeh.models.LayoutDOM | None,
    string: str,
    *,
    color: Any = ...,
    size: Any = ...,
    **artist_kws: dict,
) -> bokeh.models.LayoutDOM: ...
def get_figsize(plot_collection: Incomplete) -> None: ...
def create_plotting_grid(
    number: int,
    rows: int = ...,
    cols: int = ...,
    *,
    figsize: tuple[float, float] | None = ...,
    figsize_units: Literal["inches", "dots"] = ...,
    squeeze: bool = ...,
    sharex: bool = ...,
    sharey: bool = ...,
    polar: bool = ...,
    width_ratios: ArrayLike | None = ...,
    height_ratios: ArrayLike | None = ...,
    plot_hspace: float | None = ...,
    subplot_kws: dict | None = ...,
    **kwargs: Incomplete,
) -> tuple[
    bokeh.models.LayoutDOM | None, bokeh.models.Plot | NDArray[bokeh.models.Plot]
]: ...
def _filter_kwargs(kwargs: Incomplete, artist_kws: Incomplete) -> None: ...
def _float_or_str_size(size: Incomplete) -> None: ...
def hist(
    y: Incomplete,
    l_e: Incomplete,
    r_e: Incomplete,
    target: Incomplete,
    *,
    bottom: Incomplete = ...,
    color: Incomplete = ...,
    facecolor: Incomplete = ...,
    edgecolor: Incomplete = ...,
    alpha: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def line(
    x: Incomplete,
    y: Incomplete,
    target: Incomplete,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    width: Incomplete = ...,
    linestyle: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def multiple_lines(
    x: Incomplete,
    y: Incomplete,
    target: Incomplete,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    width: Incomplete = ...,
    linestyle: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def scatter(
    x: Incomplete,
    y: Incomplete,
    target: Incomplete,
    *,
    size: Incomplete = ...,
    marker: Incomplete = ...,
    alpha: Incomplete = ...,
    color: Incomplete = ...,
    facecolor: Incomplete = ...,
    edgecolor: Incomplete = ...,
    width: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def step(
    x: Incomplete,
    y: Incomplete,
    target: Incomplete,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    width: Incomplete = ...,
    linestyle: Incomplete = ...,
    step_mode: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def text(
    x: Incomplete,
    y: Incomplete,
    string: Incomplete,
    target: Incomplete,
    *,
    size: Incomplete = ...,
    alpha: Incomplete = ...,
    color: Incomplete = ...,
    vertical_align: Incomplete = ...,
    horizontal_align: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def fill_between_y(
    x: Incomplete,
    y_bottom: Incomplete,
    y_top: Incomplete,
    target: Incomplete,
    **artist_kws: Incomplete,
) -> None: ...
def vline(
    x: Incomplete,
    target: Incomplete,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    width: Incomplete = ...,
    linestyle: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def hline(
    y: Incomplete,
    target: Incomplete,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    width: Incomplete = ...,
    linestyle: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def vspan(
    xmin: Incomplete,
    xmax: Incomplete,
    target: Incomplete,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def hspan(
    ymin: Incomplete,
    ymax: Incomplete,
    target: Incomplete,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def ciliney(
    x: Incomplete,
    y_bottom: Incomplete,
    y_top: Incomplete,
    target: Incomplete,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    width: Incomplete = ...,
    linestyle: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def title(
    string: Incomplete,
    target: Incomplete,
    *,
    size: Incomplete = ...,
    color: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def ylabel(
    string: Incomplete,
    target: Incomplete,
    *,
    size: Incomplete = ...,
    color: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def xlabel(
    string: Incomplete,
    target: Incomplete,
    *,
    size: Incomplete = ...,
    color: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def xticks(
    ticks: Incomplete,
    labels: Incomplete,
    target: Incomplete,
    *,
    rotation: Incomplete = ...,
    color: Incomplete = ...,
    size: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def yticks(
    ticks: Incomplete,
    labels: Incomplete,
    target: Incomplete,
    *,
    rotation: Incomplete = ...,
    color: Incomplete = ...,
    size: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def xlim(lims: Incomplete, target: Incomplete, **artist_kws: Incomplete) -> None: ...
def ylim(lims: Incomplete, target: Incomplete, **artist_kws: Incomplete) -> None: ...
def ticklabel_props(
    target: Incomplete,
    *,
    axis: Incomplete = ...,
    size: Incomplete = ...,
    color: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def set_ticklabel_visibility(
    target: Incomplete, *, axis: Incomplete = ..., visible: Incomplete = ...
) -> None: ...
def remove_ticks(target: Incomplete, *, axis: Incomplete = ...) -> None: ...
def remove_axis(target: Incomplete, axis: Incomplete = ...) -> None: ...
def xscale(target: Incomplete, scale: Incomplete) -> None: ...
def yscale(target: Incomplete, scale: Incomplete) -> None: ...
def grid(target: Incomplete, axis: Incomplete, color: Incomplete) -> None: ...
def contour(
    x: ArrayLike,
    y: ArrayLike,
    density: ArrayLike,
    target: bokeh.models.Plot,
    *,
    levels: ArrayLike | None = ...,
    color: Any = ...,
    alpha: Any = ...,
    **artist_kws: Incomplete,
) -> bokeh.models.GlyphRenderer: ...
def contourf(
    x: ArrayLike,
    y: ArrayLike,
    density: ArrayLike,
    target: bokeh.models.Plot,
    *,
    levels: ArrayLike | None = ...,
    color: Any = ...,
    alpha: Any = ...,
    cmap: str | None = ...,
    **artist_kws: Incomplete,
) -> bokeh.models.GlyphRenderer: ...
