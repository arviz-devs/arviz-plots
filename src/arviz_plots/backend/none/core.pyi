# File generated with docstub

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray

ALLOW_KWARGS: Incomplete

class UnsetDefault:
    def __repr__(self) -> None: ...

unset: Incomplete

def get_background_color() -> None: ...
def get_default_aes(
    aes_key: str, n: int, kwargs: Mapping[str, ArrayLike] | None = ...
) -> NDArray: ...
def scale_fig_size(
    figsize: tuple[float, float] | None,
    rows: int = ...,
    cols: int = ...,
    figsize_units: Literal["inches", "dots"] | None = ...,
) -> tuple[float, float]: ...
def show(figure: Any) -> None: ...
def savefig(figure: Any, path: Path, **kwargs: dict) -> None: ...
def set_figure_title(
    figure: dict, string: str, *, color: Any = ..., size: Any = ..., **artist_kws: dict
) -> dict: ...
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
    height_ratios: Incomplete = ...,
    plot_hspace: float | None = ...,
    subplot_kws: Mapping | None = ...,
    **kwargs: Mapping,
) -> tuple[False, list | NDArray[object]]: ...
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
    x: ArrayLike,
    y: ArrayLike,
    target: Any,
    *,
    color: Incomplete = ...,
    alpha: Incomplete = ...,
    width: Incomplete = ...,
    linestyle: Incomplete = ...,
    **artist_kws: Incomplete,
) -> Any: ...
def multiple_lines(
    x: ArrayLike,
    y: ArrayLike,
    target: list[Any],
    *,
    color: Any = ...,
    alpha: Any = ...,
    width: Any = ...,
    linestyle: Any = ...,
    **artist_kws: Incomplete,
) -> dict: ...
def scatter(
    x: ArrayLike,
    y: ArrayLike,
    target: Any,
    *,
    size: Any = ...,
    marker: Any = ...,
    alpha: Any = ...,
    color: Any = ...,
    facecolor: Any = ...,
    edgecolor: Any = ...,
    width: Any = ...,
    **artist_kws: Incomplete,
) -> Any: ...
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
    x: ArrayLike,
    y: ArrayLike,
    density: ArrayLike,
    target: list,
    *,
    levels: ArrayLike | None = ...,
    color: Any = ...,
    alpha: Any = ...,
    **artist_kws: Incomplete,
) -> dict: ...
def contourf(
    x: ArrayLike,
    y: ArrayLike,
    density: ArrayLike,
    target: list,
    *,
    levels: ArrayLike | None = ...,
    color: Any = ...,
    alpha: Any = ...,
    cmap: Incomplete = ...,
    **artist_kws: Incomplete,
) -> dict: ...
