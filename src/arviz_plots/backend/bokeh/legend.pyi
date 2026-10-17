# File generated with docstub

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

from _typeshed import Incomplete
from bokeh.models import Legend

from arviz_plots.backend.bokeh.core import expand_aesthetic_aliases

from .plot_collection import PlotCollection

def dealiase_line_kwargs(**kwargs: Incomplete) -> None: ...
def legend(
    plot_collection: PlotCollection,
    kwarg_list: Sequence[Mapping],
    label_list: Sequence[str],
    title: str | None = ...,
    visual_type: Literal["line", "scatter", "rectangle"] = ...,
    visual_kwargs: Mapping | None = ...,
    legend_dim: str | Sequence[str] | None = ...,
    update_visuals: bool = ...,
    legend_target: tuple[int, int] | None = ...,
    side: str = ...,
    **kwargs: Incomplete,
) -> None: ...
