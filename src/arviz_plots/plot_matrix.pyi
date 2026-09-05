# File generated with docstub

from collections.abc import Callable, Hashable, Mapping
from importlib import import_module

import numpy as np
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams, xarray_sel_iter
from xarray import Dataset, DataTree

from arviz_plots.plot_collection import (
    PlotCollection,
    concat_model_dict,
    process_kwargs_subset,
)

def subset_matrix_da(
    da: Incomplete,
    var_name_x: Hashable,
    selection_x: Mapping,
    var_name_y: Hashable | None = ...,
    selection_y: Mapping | None = ...,
    return_dataarray: bool = ...,
) -> None: ...

class PlotMatrix(PlotCollection):
    viz: DataTree
    aes: DataTree

    def __init__(
        self,
        data: Dataset,
        facet_dims: list[Hashable],
        aes: Mapping[str, list[Hashable]] | None = ...,
        backend: str | None = ...,
        figure_kwargs: Incomplete = ...,
        **kwargs: Mapping,
    ) -> None: ...
    @property
    def facet_dims(self) -> None: ...
    def _generate_viz_dt(self, **figure_kwargs: Incomplete) -> None: ...
    def get_target(
        self,
        var_name: Hashable,
        selection: Mapping,
        var_name_y: Hashable | None = ...,
        selection_y: Mapping | None = ...,
    ) -> None: ...
    def allocate_artist(
        self,
        func_label: Incomplete,
        data: Incomplete,
        all_loop_dims: Incomplete,
        dim_to_idx: Incomplete = ...,
        artist_dims: Incomplete = ...,
        ignore_aes: Incomplete = ...,
    ) -> None: ...
    def store_in_artist_da(
        self,
        aux_artist: Incomplete,
        func_label: Incomplete,
        var_name: Hashable,
        sel: Mapping,
        var_name_y: Hashable | None = ...,
        sel_y: Mapping | None = ...,
    ) -> None: ...
    def map_upper(self, *args: Incomplete, **kwargs: Incomplete) -> None: ...
    def map_lower(self, *args: Incomplete, **kwargs: Incomplete) -> None: ...
    def map_triangle(
        self,
        func: Callable,
        func_label: str | None = ...,
        *,
        data: Dataset | None = ...,
        loop_data: Dataset | str | None = ...,
        triangle: Incomplete = ...,
        coords: Mapping | None = ...,
        ignore_aes: set = ...,
        subset_info: bool = ...,
        store_artist: bool = ...,
        artist_dims: Mapping[Hashable, int] | None = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def map(
        self,
        func: Callable,
        func_label: str | None = ...,
        *,
        data: Dataset | None = ...,
        coords: Mapping | None = ...,
        ignore_aes: set = ...,
        subset_info: bool = ...,
        store_artist: bool = ...,
        artist_dims: Mapping[Hashable, int] | None = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def set_fixed_var_attributes(
        self, index: Incomplete, orientation: Incomplete = ...
    ) -> None: ...
    def map_row(
        self,
        func: Callable,
        func_label: str | None = ...,
        index: int = ...,
        *,
        data: Dataset | None = ...,
        coords: Mapping | None = ...,
        ignore_aes: str | set[str] = ...,
        subset_info: bool = ...,
        store_artist: bool = ...,
        artist_dims: Mapping[Hashable, int] | None = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def map_col(
        self,
        func: Callable,
        func_label: str | None = ...,
        index: int = ...,
        *,
        data: Dataset | None = ...,
        coords: Mapping | None = ...,
        ignore_aes: str | set[str] = ...,
        subset_info: bool = ...,
        store_artist: bool = ...,
        artist_dims: Mapping[Hashable, int] | None = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    @property
    def viz(self) -> None: ...
