# File generated with docstub

import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import arviz_base
import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from arviz_base import rcParams, xarray_sel_iter
from arviz_base.labels import BaseLabeller, NoVarLabeller
from arviz_base.utils import _var_names
from xarray import Dataset, DataTree

def backend_from_object(
    obj: Incomplete, return_module: bool = ...
) -> ModuleType | str: ...
def concat_model_dict(data: Incomplete) -> None: ...
def sel_subset(sel: Incomplete, ds_da: Incomplete) -> None: ...
def subset_ds(ds: Dataset, var_name: Hashable, sel: Mapping) -> None: ...
def try_da_subset(da: Incomplete, sel: Incomplete) -> None: ...
def process_kwargs_subset(
    value: Incomplete, var_name: Incomplete, sel: Incomplete
) -> None: ...
def process_facet_dims(data: Incomplete, facet_dims: Incomplete) -> None: ...
def leaf_dataset(dt: DataTree, leaf_name: Hashable) -> Dataset: ...

class PlotCollection:
    viz: DataTree
    aes: DataTree

    def __init__(
        self,
        data: Dataset,
        viz_dt: DataTree,
        aes_dt: DataTree | None = ...,
        aes: Mapping[str, list[Hashable]] | None = ...,
        backend: str | None = ...,
        **kwargs: Mapping,
    ) -> None: ...
    def _repr_html_(self) -> None: ...
    def _display_(self) -> None: ...
    @property
    def aes(self) -> None: ...
    @aes.setter
    def aes(self, value: Incomplete) -> None: ...
    @property
    def viz(self) -> None: ...
    @viz.setter
    def viz(self, value: Incomplete) -> None: ...
    @property
    def coords(self) -> None: ...
    @coords.setter
    def coords(self, value: Incomplete) -> None: ...
    @property
    def data(self) -> None: ...
    @data.setter
    def data(self, value: Incomplete) -> None: ...
    @property
    def aes_set(self) -> None: ...
    def show(self) -> None: ...
    def savefig(self, filename: str | Path, **kwargs: Incomplete) -> None: ...
    def generate_aes_dt(
        self,
        aes: Mapping[str, list[Hashable] | False],
        data: Dataset | None = ...,
        **kwargs: Mapping,
    ) -> None: ...
    def get_aes_as_dataset(self, aes_key: str) -> Dataset: ...
    def update_aes_from_dataset(self, aes_key: str, dataset: Dataset) -> None: ...
    @property
    def facet_dims(self) -> None: ...
    def get_viz(
        self,
        artist_name: str,
        var_name: str | None = ...,
        sel: Mapping | None = ...,
        **sel_kwargs: Mapping,
    ) -> None: ...
    def rename_visuals(
        self, name_dict: Mapping | None = ..., **names: Mapping
    ) -> None: ...
    @classmethod
    def wrap(
        cls,
        data: Dataset | dict[str, Dataset],
        cols: Iterable[Hashable] | None = ...,
        col_wrap: int | None = ...,
        backend: str | None = ...,
        figure_kwargs: Mapping | None = ...,
        **kwargs: Mapping,
    ) -> None: ...
    @classmethod
    def grid(
        cls,
        data: Dataset | dict[str, Dataset],
        cols: Iterable[Hashable] | None = ...,
        rows: Iterable[Hashable] | None = ...,
        backend: str | None = ...,
        figure_kwargs: Mapping | None = ...,
        **kwargs: Mapping,
    ) -> None: ...
    def update_aes(
        self, ignore_aes: Incomplete = ..., coords: Incomplete = ...
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
    def get_target(self, var_name: Incomplete, selection: Incomplete) -> None: ...
    def iget_target(self, row_index: int, col_index: int) -> None: ...
    def get_aes_kwargs(self, aes: list, var_name: str, selection: dict) -> dict: ...
    def map(
        self,
        func: Callable,
        func_label: str | None = ...,
        *,
        data: Dataset | xarray.DataArray | None = ...,
        coords: Mapping | None = ...,
        ignore_aes: set | Literal["all"] = ...,
        subset_info: bool = ...,
        store_artist: bool = ...,
        artist_dims: Mapping[Hashable, int] | None = ...,
        **kwargs: Mapping,
    ) -> None: ...
    def store_in_artist_da(
        self,
        aux_artist: Incomplete,
        func_label: Incomplete,
        var_name: Incomplete,
        sel: Incomplete,
    ) -> None: ...
    def add_title(
        self, text: str, *, color: Any = ..., size: Any | None = ..., **kwargs: Mapping
    ) -> None: ...
    def facet_map(
        self,
        func: str | Callable,
        func_label: str | None = ...,
        *,
        var_names: str | list[str] | None = ...,
        filter_vars: Literal[None, "like", "regex"] | None = ...,
        coords: Mapping | None = ...,
        **kwargs: Incomplete,
    ) -> PlotCollection: ...
    def add_legend(
        self,
        dim: Hashable | Iterable[Hashable],
        aes: str | Iterable[str] | None = ...,
        visual_kwargs: Mapping | None = ...,
        title: str | None = ...,
        text_only: bool = ...,
        labeller: arviz_base.labels.Labeller | None = ...,
        **kwargs: Mapping,
    ) -> object: ...
