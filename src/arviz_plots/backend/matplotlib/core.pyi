# File generated with docstub

import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import numpy as np
from _typeshed import Incomplete
from matplotlib import ticker
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.pyplot import rcParams
from matplotlib.pyplot import show as _show
from matplotlib.pyplot import subplots
from matplotlib.text import Text

from arviz_plots.backend.alias_utils import create_aesthetic_handlers
from arviz_plots.backend.none import get_default_aes as get_agnostic_default_aes

class UnsetDefault:
    pass

unset: Incomplete

class SquareRootBaseScale(mscale.ScaleBase):

    name: Incomplete

    def set_default_locators_and_formatters(self, axis: Incomplete) -> None: ...
    def limit_range_for_scale(
        self, vmin: Incomplete, vmax: Incomplete, minpos: Incomplete
    ) -> None: ...

    class SquareRootTransform(mtransforms.Transform):

        input_dims: Incomplete
        output_dims: Incomplete
        is_separable: Incomplete

        def transform_non_affine(self, values: Incomplete) -> None: ...
        def inverted(self) -> None: ...

    class InvertedSquareRootTransform(mtransforms.Transform):

        input_dims: Incomplete
        output_dims: Incomplete
        is_separable: Incomplete

        def transform(self, values: Incomplete) -> None: ...
        def inverted(self) -> None: ...

    def get_transform(self) -> None: ...

class SquareRootScale(SquareRootBaseScale):
    def __init__(self, axis: Incomplete) -> None: ...

class SquareRootScale311(SquareRootBaseScale):
    def __init__(self) -> None: ...

try:
    pass
except TypeError:
    pass

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
def savefig(figure: matplotlib.figure.Figure, path: Path, **kwargs: dict) -> None: ...
def set_figure_title(
    figure: matplotlib.figure.Figure,
    string: str,
    *,
    color: Any = ...,
    size: Any = ...,
    **artist_kws: dict,
) -> matplotlib.figure.Figure: ...
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
    width_ratios: list | None = ...,
    height_ratios: list | None = ...,
    plot_hspace: float | None = ...,
    subplot_kws: dict | None = ...,
    **kwargs: dict,
) -> matplotlib.figure.Figure: ...
def _filter_kwargs(
    kwargs: Incomplete, visual: Incomplete, artist_kws: Incomplete
) -> None: ...
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
    y_max: Incomplete,
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
def set_ticklabel_visibility(
    target: Incomplete, *, axis: Incomplete = ..., visible: Incomplete = ...
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
def remove_ticks(target: Incomplete, *, axis: Incomplete = ...) -> None: ...
def remove_axis(target: Incomplete, axis: Incomplete = ...) -> None: ...
def xscale(target: Incomplete, scale: Incomplete) -> None: ...
def yscale(target: Incomplete, scale: Incomplete) -> None: ...
def grid(target: Incomplete, axis: Incomplete, color: Incomplete) -> None: ...
def contour(
    x: Incomplete,
    y: Incomplete,
    density: Incomplete,
    target: Incomplete,
    *,
    levels: Incomplete = ...,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
def contourf(
    x: Incomplete,
    y: Incomplete,
    density: Incomplete,
    target: Incomplete,
    *,
    levels: Incomplete = ...,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    **artist_kws: Incomplete,
) -> None: ...
