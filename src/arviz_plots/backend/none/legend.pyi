# File generated with docstub

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from _typeshed import Incomplete

from .plot_collection import PlotCollection

def legend(
    plot_collection: PlotCollection,
    kwarg_list: Sequence[Mapping],
    label_list: Sequence[str],
    title: str | None = ...,
    visual_type: Literal["line", "scatter", "rectangle"] = ...,
    visual_kwargs: Mapping | None = ...,
    legend_dim: str | Sequence[str] | None = ...,
    update_visuals: bool | None = ...,
    **kwargs: Incomplete,
) -> Any: ...
