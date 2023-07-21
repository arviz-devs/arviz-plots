"""Plot collection classes."""
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base.sel_utils import xarray_sel_iter
from datatree import DataTree


def sel_subset(sel, present_dims):
    return {key: value for key, value in sel.items() if key in present_dims}


def subset_ds(ds, var_name, sel):
    subset_dict = sel_subset(sel, ds[var_name].dims)
    if subset_dict:
        out = ds[var_name].sel(subset_dict)
    else:
        out = ds[var_name]
    if out.size == 1:
        return out.item()
    return out.values


def _process_facet_dims(data, facet_dims):
    if not facet_dims:
        return 1, {}
    facets_per_var = {}
    if "__variable__" in facet_dims:
        for var_name, da in data.items():
            lenghts = [len(da[dim]) for dim in facet_dims if dim in da.dims]
            facets_per_var[var_name] = np.prod(lenghts) if lenghts else 1
        n_facets = np.sum(list(facets_per_var.values()))
    else:
        missing_dims = {
            var_name: [dim for dim in facet_dims if dim not in da.dims]
            for var_name, da in data.items()
        }
        missing_dims = {k: v for k, v in missing_dims.items() if v}
        if any(missing_dims.values()):
            raise ValueError(
                "All variables must have all facetting dimensions, but found the following "
                f"dims to be missing in these variables: {missing_dims}"
            )
        n_facets = np.prod([data.sizes[dim] for dim in facet_dims])
    return n_facets, facets_per_var


class PlotMuseum:
    def __init__(self, data, viz_dt, aes_dt=None, aes=None, backend=None, **kwargs):

        self._data = data
        self.viz = viz_dt
        self.aes = aes_dt

        if backend is not None:
            self.backend = backend

        if aes is None:
            aes = {}

        self._aes = aes
        self._kwargs = kwargs

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        # might want/be possible to make some checks on the data before setting it
        self._data = value


    def show(self):
        if "chart" not in self.viz:
            raise ValueError("No plot found to be shown")
        plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")
        plot_bknd.show(self.viz["chart"].item())

    def generate_aes_dt(self, aes, **kwargs):
        if aes is None:
            aes = {}
        self._aes = aes
        self._kwargs = kwargs
        self.aes = DataTree()
        for var_name, da in self.data.items():
            ds = xr.Dataset()
            for aes_key, dims in aes.items():
                aes_vals = kwargs.get(aes_key, [None])
                aes_raw_shape = [da.sizes[dim] for dim in dims if dim in da.dims]
                if not aes_raw_shape:
                    ds[aes_key] = aes_vals[0]
                    continue
                n_aes = np.prod(aes_raw_shape)
                n_aes_vals = len(aes_vals)
                if n_aes_vals > n_aes:
                    aes_vals = aes_vals[:n_aes]
                elif n_aes_vals < n_aes:
                    aes_vals = np.tile(aes_vals, (n_aes // n_aes_vals) + 1)[:n_aes]
                ds[aes_key] = xr.DataArray(
                    np.array(aes_vals).reshape(aes_raw_shape),
                    dims=dims,
                    coords={dim: da.coords[dim] for dim in dims if dim in da.coords},
                )
            DataTree(name=var_name, parent=self.aes, data=ds)

    @property
    def base_loop_dims(self):
        if "plot" in self.viz.data_vars:
            return set(self.viz["plot"].dims)
        return set(dim for da in self.viz.children.values() for dim in da["plot"].dims)

    def get_viz(self, var_name):
        return self.viz if "plot" in self.viz.data_vars else self.viz[var_name]

    @classmethod
    def wrap(
        cls,
        data,
        cols=None,
        col_wrap=4,
        backend="matplotlib",
        plot_grid_kws=None,
        **kwargs,
    ):
        if cols is None:
            cols = []
        if plot_grid_kws is None:
            plot_grid_kws = {}

        n_plots, plots_per_var = _process_facet_dims(data, cols)
        if n_plots <= col_wrap:
            n_rows, n_cols = 1, n_plots
        else:
            div_mod = divmod(n_plots, col_wrap)
            n_rows = div_mod[0] + (div_mod[1] != 0)
            n_cols = col_wrap

        plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
        fig, ax_ary = plot_bknd.create_plotting_grid(
            n_plots, n_rows, n_cols, squeeze=False, **plot_grid_kws
        )
        col_id, row_id = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        viz_dict = {}
        flat_ax_ary = ax_ary.flatten()[:n_plots]
        flat_row_id = row_id.flatten()[:n_plots]
        flat_col_id = col_id.flatten()[:n_plots]
        if "__variable__" not in cols:
            dims = cols  # use provided dim orders, not existing ones
            plots_raw_shape = [data.sizes[dim] for dim in dims]
            viz_dict["/"] = xr.Dataset(
                {
                    "chart": fig,
                    "plot": (dims, flat_ax_ary.reshape(plots_raw_shape)),
                    "row": (dims, flat_row_id.reshape(plots_raw_shape)),
                    "col": (dims, flat_col_id.reshape(plots_raw_shape)),
                },
                coords={dim: data[dim] for dim in dims},
            )
        else:
            viz_dict["/"] = xr.Dataset({"chart": xr.DataArray(fig)})
            all_dims = cols
            facet_cumulative = 0
            for var_name, da in data.items():
                dims = [dim for dim in all_dims if dim in da.dims]
                plots_raw_shape = [data.sizes[dim] for dim in dims]
                col_slice = (
                    slice(None, None)
                    if var_name not in plots_per_var
                    else slice(facet_cumulative, facet_cumulative + plots_per_var[var_name])
                )
                facet_cumulative += plots_per_var[var_name]
                viz_dict[var_name] = xr.Dataset(
                    {
                        "plot": (
                            dims,
                            flat_ax_ary[col_slice].reshape(plots_raw_shape),
                        ),
                        "row": (
                            dims,
                            flat_row_id[col_slice].reshape(plots_raw_shape),
                        ),
                        "col": (
                            dims,
                            flat_col_id[col_slice].reshape(plots_raw_shape),
                        ),
                    }
                )
        viz_dt = DataTree.from_dict(viz_dict)
        return cls(data, viz_dt, backend=backend, **kwargs)

    @classmethod
    def grid(
        cls,
        data,
        cols=None,
        rows=None,
        backend="matplotlib",
        plot_grid_kws=None,
        **kwargs,
    ):
        if cols is None:
            cols = []
        if rows is None:
            rows = []
        if plot_grid_kws is None:
            plot_grid_kws = {}
        repeated_dims = [col for col in cols if col in rows]
        if repeated_dims:
            raise ValueError("The same dimension can't be used for both cols and rows.")

        n_cols, cols_per_var = _process_facet_dims(data, cols)
        n_rows, rows_per_var = _process_facet_dims(data, rows)

        n_plots = n_cols * n_rows
        plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
        fig, ax_ary = plot_bknd.create_plotting_grid(
            n_plots, n_rows, n_cols, squeeze=False, **plot_grid_kws
        )
        col_id, row_id = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        viz_dict = {}
        if "__variable__" not in cols and "__variable__" not in rows:
            dims = tuple((*rows, *cols))  # use provided dim orders, not existing ones
            plots_raw_shape = [data.sizes[dim] for dim in dims]
            viz_dict["/"] = xr.Dataset(
                {
                    "chart": fig,
                    "plot": (dims, ax_ary.flatten().reshape(plots_raw_shape)),
                    "row": (dims, row_id.flatten().reshape(plots_raw_shape)),
                    "col": (dims, col_id.flatten().reshape(plots_raw_shape)),
                },
                coords={dim: data[dim] for dim in dims},
            )
        else:
            viz_dict["/"] = xr.Dataset({"chart": xr.DataArray(fig)})
            all_dims = tuple((*rows, *cols))  # use provided dim orders, not existing ones
            facet_cumulative = 0
            for var_name, da in data.items():
                dims = [dim for dim in all_dims if dim in da.dims]
                plots_raw_shape = [data.sizes[dim] for dim in dims]
                row_slice = (
                    slice(None, None)
                    if var_name not in rows_per_var
                    else slice(facet_cumulative, facet_cumulative + rows_per_var[var_name])
                )
                col_slice = (
                    slice(None, None)
                    if var_name not in cols_per_var
                    else slice(facet_cumulative, facet_cumulative + cols_per_var[var_name])
                )
                if rows_per_var:
                    facet_cumulative += rows_per_var[var_name]
                else:
                    facet_cumulative += cols_per_var[var_name]
                viz_dict[var_name] = xr.Dataset(
                    {
                        "plot": (
                            dims,
                            ax_ary[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                        ),
                        "row": (
                            dims,
                            row_id[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                        ),
                        "col": (
                            dims,
                            col_id[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                        ),
                    }
                )
        viz_dt = DataTree.from_dict(viz_dict)
        return cls(data, viz_dt, backend=backend, **kwargs)

    def update_aes(self, ignore_aes=frozenset(), coords=None):
        if coords is None:
            coords = {}
        aes = [aes_key for aes_key in self._aes.keys() if aes_key not in ignore_aes]
        aes_dims = [dim for aes_key in aes for dim in self._aes[aes_key]]
        all_loop_dims = self.base_loop_dims.union(aes_dims).difference(coords.keys())
        return aes, all_loop_dims

    def allocate_artist(self, fun_label, data, all_loop_dims, artist_dims=None):
        if artist_dims is None:
            artist_dims = {}
        for var_name, da in data.items():
            if var_name not in self.viz.children:
                DataTree(name=var_name, parent=self.viz)
            inherited_dims = [dim for dim in da.dims if dim in all_loop_dims]
            artist_shape = [da.sizes[dim] for dim in inherited_dims] + list(
                artist_dims.values()
            )
            all_artist_dims = inherited_dims + list(artist_dims.keys())

            self.viz[var_name][fun_label] = xr.DataArray(
                np.empty(artist_shape, dtype=object),
                dims=all_artist_dims,
                coords={dim: data[dim] for dim in inherited_dims},
            )
    def get_target(self, var_name, selection):
        return subset_ds(self.get_viz(var_name), "plot", selection)

    def get_aes_kwargs(self, aes, var_name, selection):
        aes_kwargs = {}
        for aes_key in aes:
            aes_kwargs[aes_key] = subset_ds(self.aes[var_name], aes_key, selection)
        return aes_kwargs

    def plot_iterator(self, ignore_aes=frozenset(), coords=None):
        if coords is None:
            coords = {}
        if self.aes is None:
            self.generate_aes_dt(self._aes, **self._kwargs)
        aes, all_loop_dims = self.update_aes(ignore_aes, coords)
        plotters = xarray_sel_iter(
            self.data, skip_dims={dim for dim in self.data.dims if dim not in all_loop_dims}
        )
        for var_name, sel, isel in plotters:
            sel_plus = {**sel, **coords}
            target = self.get_target(var_name, sel_plus)
            aes_kwargs = self.get_aes_kwargs(aes, var_name, sel_plus)
            yield target, var_name, sel, isel, aes_kwargs

    def map(
        self,
        fun,
        fun_label=None,
        *,
        data=None,
        coords=None,
        ignore_aes=frozenset(),
        subset_info=False,
        store_artist=True,
        artist_dims=None,
        **kwargs,
    ):
        if coords is None:
            coords = {}
        if self.aes is None:
            self.generate_aes_dt(self._aes, **self._kwargs)
        if fun_label is None:
            fun_label = fun.__name__

        data = self.data if data is None else data

        aes, all_loop_dims = self.update_aes(ignore_aes, coords)
        plotters = xarray_sel_iter(
            data, skip_dims={dim for dim in data.dims if dim not in all_loop_dims}
        )
        if store_artist:
            self.allocate_artist(fun_label=fun_label, data=data, all_loop_dims=all_loop_dims, artist_dims=artist_dims)

        for var_name, sel, isel in plotters:
            da = data[var_name].sel(sel)
            sel_plus = {**sel, **coords}
            target = self.get_target(var_name, sel_plus)

            aes_kwargs = self.get_aes_kwargs(aes, var_name, sel_plus)

            fun_kwargs = {**aes_kwargs, **kwargs}
            fun_kwargs["backend"] = self.backend
            if subset_info:
                fun_kwargs = {**fun_kwargs, "var_name": var_name, "sel": sel, "isel": isel}
            aux_artist = fun(da, target=target, **fun_kwargs)
            if store_artist:
                self.viz[var_name][fun_label].loc[sel] = aux_artist

    def add_legend(self, artist, **kwargs):
        raise NotImplementedError()
