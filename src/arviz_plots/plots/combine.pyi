# File generated with docstub

from collections.abc import Callable, Hashable, Mapping, Sequence
from importlib import import_module
from typing import Literal

from _typeshed import Incomplete
from arviz_base import rcParams
from arviz_base.validate import validate_sample_dims
from xarray import DataTree

from arviz_plots import PlotCollection
from arviz_plots.plot_collection import backend_from_object
from arviz_plots.plots.utils import process_group_variables_coords, set_grid_layout

from .plot_collection import PlotCollection

def render(da: Incomplete, target: Incomplete, **kwargs: Incomplete) -> None: ...
def combine_plots(
    dt: DataTree | dict[str, DataTree],
    plots: list[tuple[Callable, Mapping]],
    var_names: str | Sequence[str] | None = ...,
    filter_vars: Literal[None, "like", "regex"] | None = ...,
    group: str = ...,
    coords: dict | None = ...,
    sample_dims: str | Sequence[Hashable] | None = ...,
    expand: Literal["column", "row"] = ...,
    plot_names: list[str] | None = ...,
    backend: Literal["matplotlib", "bokeh", "plotly"] | None = ...,
    **pc_kwargs: Incomplete,
) -> PlotCollection: ...
