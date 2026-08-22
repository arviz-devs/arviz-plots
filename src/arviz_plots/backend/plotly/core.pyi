# File generated with docstub

import inspect
import math
import re
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
from _typeshed import Incomplete
from numpy.typing import ArrayLike
from plotly.graph_objects import Scatter
from plotly.subplots import make_subplots
from webcolors import hex_to_rgb, name_to_hex, name_to_rgb

from arviz_plots.backend.alias_utils import create_aesthetic_handlers
from arviz_plots.backend.none import get_default_aes as get_agnostic_default_aes

class UnsetDefault:
    pass

unset: Incomplete
pat: Incomplete

def is_shared_x(fig: Incomplete) -> None: ...
def apply_square_root_scale(plotly_plot: Incomplete, axis: Incomplete) -> None: ...
def str_to_plotly_html(string: Incomplete) -> None: ...
def combine_color_alpha(color: Incomplete, alpha: Incomplete = ...) -> None: ...
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
def get_figsize(plot_collection: Incomplete) -> None: ...
def remove_row_col_from_doc(docstring: Incomplete) -> None: ...

class PlotlyPlot:
    def __init__(
        self, figure: Incomplete, row: Incomplete, col: Incomplete
    ) -> None: ...
    def __getattr__(self, name: Incomplete) -> None: ...

def show(figure: Incomplete) -> None: ...
def savefig(
    figure: plotly.graph_objects.Figure, path: Path, **kwargs: dict
) -> None: ...
def set_figure_title(
    figure: plotly.graph_objects.Figure,
    string: str,
    *,
    color: Any = ...,
    size: Any = ...,
    **artist_kws: dict,
) -> plotly.graph_objects.Figure: ...
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
    width_ratios: list | None = ...,
    height_ratios: list | None = ...,
    plot_hspace: float | None = ...,
    subplot_kws: dict | None = ...,
    **kwargs: dict,
) -> plotly.graph_objects.Figure | None: ...
def _filter_kwargs(kwargs: Incomplete, artist_kws: Incomplete) -> None: ...
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
    x: ArrayLike,
    y: ArrayLike,
    target: PlotlyPlot,
    *,
    color: Any = ...,
    alpha: Any = ...,
    width: Any = ...,
    linestyle: Any = ...,
    **artist_kws: Incomplete,
) -> Scatter: ...
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
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
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
def set_ticklabel_visibility(
    target: Incomplete, *, axis: Incomplete = ..., visible: Incomplete = ...
) -> None: ...
def ylim(lims: Incomplete, target: Incomplete, **artist_kws: Incomplete) -> None: ...
def xscale(target: Incomplete, scale: Incomplete) -> None: ...
def yscale(target: Incomplete, scale: Incomplete) -> None: ...
def ticklabel_props(
    target: Incomplete,
    *,
    axis: Incomplete = ...,
    size: Incomplete = ...,
    color: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def remove_ticks(target: Incomplete, *, axis: Incomplete = ...) -> None: ...
def remove_axis(target: Incomplete, axis: Incomplete = ...) -> None: ...
def grid(
    target: Incomplete,
    axis: Incomplete = ...,
    color: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def contour(
    x: ArrayLike,
    y: ArrayLike,
    density: ArrayLike,
    target: PlotlyPlot,
    *,
    levels: ArrayLike | None = ...,
    color: Any = ...,
    alpha: Any = ...,
    **artist_kws: Incomplete,
) -> list[go.Contour]: ...
def contourf(
    x: ArrayLike,
    y: ArrayLike,
    density: ArrayLike,
    target: PlotlyPlot,
    *,
    levels: ArrayLike | None = ...,
    color: Any = ...,
    alpha: Any = ...,
    cmap: str | None = ...,
    **artist_kws: Incomplete,
) -> go.Contour: ...
