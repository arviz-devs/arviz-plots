# pylint: disable=too-many-lines, too-many-public-methods
"""Plot matrix class."""
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import rcParams, xarray_sel_iter

from arviz_plots.plot_collection import PlotCollection, concat_model_dict, process_facet_dims


def subset_matrix_da(
    da, var_name_x, selection_x, var_name_y=None, selection_y=None, return_dataarray=False
):
    if var_name_y is None:
        var_name_y = var_name_x
    if selection_y is None:
        selection_y = selection_x
    out = (
        da.set_xindex("var_name_x")
        .set_xindex("var_name_y")
        .sel(var_name_x=var_name_x, var_name_y=var_name_y)
    )
    for dim in set(selection_x).union(selection_y):
        coords = {}
        dim_x = f"{dim}_x"
        dim_y = f"{dim}_y"
        if not (dim_x in da.coords and dim_y in da.coords):
            continue
        if dim in selection_x:
            coords[dim_x] = selection_x[dim]
            out = out.set_xindex(dim_x)
        if dim in selection_y:
            coords[dim_y] = selection_y[dim]
            out = out.set_xindex(dim_y)
        if coords:
            out = out.sel(coords)
    if return_dataarray:
        return out
    if out.size == 1:
        return out.item()
    return out.values


class PlotMatrix(PlotCollection):
    """Low level base class for pairwise matrix arranges of plots.

    Attributes
    ----------
    viz : DataTree
    aes : DataTree
    """

    def __init__(self, data, facet_dims, aes=None, backend=None, plot_grid_kws=None, **kwargs):
        """Initialize a PlotMatrix.

        Parameters
        ----------
        data : Dataset
            Data for which to generate the requested matrix layout of plots.
        facet_dims : list of hashable
            List of dimensions to use for facetting. It also support the ``__variable__``
            indicator to facet across variables.
        aes : mapping of {str : list of hashable}, optional
            Dictionary with :term:`aesthetics` as keys and as values a list
            of the dimensions it should be mapped to.
            See :meth:`~arviz_plots.PlotCollection.generate_aes_dt` for more details.
        backend : str, optional
            Plotting backend. It will be stored and passed down to the plotting
            functions when using methods like :meth:`~arviz_plots.PlotCollection.map`.
        **kwargs : mapping, optional
            Dictionary with :term:`aesthetics` as keys and as values a list
            of the values that should be taken by that aesthetic.

        """
        self._data = concat_model_dict(data)
        self._coords = None
        self._facet_dims = facet_dims

        if backend is None:
            backend = rcParams["plot.backend"]
        self.backend = backend

        if aes is None:
            aes = {}
        self._aes = aes
        self._kwargs = kwargs
        if plot_grid_kws is None:
            plot_grid_kws = {}

        self._viz_dt = self._generate_viz_dt(**plot_grid_kws)
        self.generate_aes_dt()

    @property
    def base_loop_dims(self):
        return set(dim for dim in self._facet_dims if dim != "__variable__")

    def _generate_viz_dt(self, **plot_grid_kws):
        data = self._data
        facet_dims = self._facet_dims
        pairs = tuple(
            xarray_sel_iter(data, skip_dims={dim for dim in data.dims if dim not in facet_dims})
        )
        n_pairs = len(pairs)
        n_plots = n_pairs**2
        plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")
        fig, ax_ary = plot_bknd.create_plotting_grid(
            n_plots, n_pairs, n_pairs, squeeze=False, **plot_grid_kws
        )
        coords = {
            "col_index": np.arange(n_pairs),
            "row_index": np.arange(n_pairs),
        }
        for dim in facet_dims:
            if dim == "__variable__":
                dim = "var_name"
                coord_values = [pair[0] for pair in pairs]
            else:
                coord_values = [pair[1].get(dim, None) for pair in pairs]
            coords[f"{dim}_x"] = (("col_index",), coord_values)
            coords[f"{dim}_y"] = (("row_index",), coord_values)
        return xr.Dataset(
            {"chart": np.array(fig, dtype=object), "plot": (("row_index", "col_index"), ax_ary)},
            coords=coords,
        )

    def get_target(self, var_name_x, selection_x, var_name_y=None, selection_y=None):
        return subset_matrix_da(
            self.viz["plot"],
            var_name_x=var_name_x,
            selection_x=selection_x,
            var_name_y=var_name_y,
            selection_y=selection_y,
        )

    def allocate_artist(self, fun_label, data, all_loop_dims, artist_dims=None, ignore_aes=None):
        if artist_dims is None:
            artist_dims = {}
        attrs = None
        if ignore_aes is not None:
            attrs = {"ignore_aes": ignore_aes}
        matrix_sizes = self.viz["plot"].sizes
        aes_dims = [
            dim for dim in data.dims if dim not in self.base_loop_dims and dim in all_loop_dims
        ]
        artist_shape = (
            list(matrix_sizes.values())
            + [data.sizes[dim] for dim in aes_dims]
            + list(artist_dims.values())
        )
        self._viz_dt[fun_label] = xr.DataArray(
            np.full(artist_shape, None, dtype=object),
            dims=list(matrix_sizes) + aes_dims + list(artist_dims.keys()),
            coords={dim: data[dim] for dim in aes_dims},
            attrs=attrs,
        )

    def store_in_artist_da(
        self, aux_artist, fun_label, var_name_x, sel_x, var_name_y=None, sel_y=None
    ):
        plot_da = subset_matrix_da(
            self.viz["plot"],
            var_name_x=var_name_x,
            selection_x=sel_x,
            var_name_y=var_name_y,
            selection_y=sel_y,
            return_dataarray=True,
        )
        self._viz_dt[fun_label].loc[
            {"row_index": plot_da["row_index"], "col_index": plot_da["col_index"]}
        ] = aux_artist

    def map_upper(self, *args, **kwargs):
        self.map_triangle(*args, triangle="upper", **kwargs)

    def map_lower(self, *args, **kwargs):
        self.map_triangle(*args, triangle="lower", **kwargs)

    def map_triangle(
        self,
        fun,
        fun_label=None,
        *,
        data=None,
        loop_data=None,
        triangle="both",
        coords=None,
        ignore_aes=frozenset(),
        subset_info=False,
        store_artist=True,
        artist_dims=None,
        **kwargs,
    ):
        if triangle not in {"lower", "upper", "both"}:
            raise ValueError("Value for triangle argument not valid")
        if coords is None:
            coords = {}
        if fun_label is None:
            fun_label = fun.__name__

        data = self.data if data is None else data
        if isinstance(loop_data, str) and loop_data == "plots":
            if "plot" in self.viz.data_vars:
                loop_data = xr.Dataset({key: self.viz.ds["plot"] for key in data.data_vars})
            else:
                loop_data = xr.Dataset(
                    {var_name: ds["plot"] for var_name, ds in self.viz.children.items()}
                )
        loop_data = data if loop_data is None else loop_data
        if not isinstance(data, xr.Dataset):
            raise TypeError("data argument must be an xarray.Dataset")

        base_loop_dims = self.base_loop_dims
        aes, all_loop_dims = self.update_aes(ignore_aes, coords)
        aes_dims = [dim for dim in all_loop_dims if dim not in base_loop_dims]
        # all variables must have all dimensions with aesthetics mapped to them
        # we only care about the dim+coord combinations
        aes_loopers = list(
            xarray_sel_iter(
                loop_data[list(loop_data.data_vars)[0]],
                skip_dims={dim for dim in loop_data.dims if dim not in aes_dims},
            )
        )
        plotters = list(
            xarray_sel_iter(
                loop_data, skip_dims={dim for dim in loop_data.dims if dim not in base_loop_dims}
            )
        )
        if store_artist:
            self.allocate_artist(
                fun_label=fun_label,
                data=loop_data,
                all_loop_dims=all_loop_dims,
                artist_dims=artist_dims,
                ignore_aes=ignore_aes,
            )
        for i, (var_name_x, sel_x_base, isel_x_base) in enumerate(plotters):
            upper_elements = plotters[:i]
            lower_elements = plotters[i + 1 :]
            if triangle == "lower":
                second_loop_elements = lower_elements
            elif triangle == "upper":
                second_loop_elements = upper_elements
            elif triangle == "both":
                second_loop_elements = lower_elements + upper_elements
            for var_name_y, sel_y_base, isel_y_base in second_loop_elements:
                da_x_base = data[var_name_x].sel(sel_x_base)
                da_y_base = data[var_name_y].sel(sel_y_base)
                for _, aes_sel, aes_isel in aes_loopers:
                    da_x = da_x_base.sel(aes_sel)
                    da_y = da_y_base.sel(aes_sel)
                    try:
                        if np.all(np.isnan(da_x)) or np.all(np.isnan(da_y)):
                            continue
                    except TypeError:
                        pass
                    sel_x = {**sel_x_base, **aes_sel}
                    sel_y = {**sel_y_base, **aes_sel}
                    isel_x = {**isel_x_base, **aes_isel}
                    isel_y = {**isel_y_base, **aes_isel}
                    sel_x_plus = {**sel_x, **coords}
                    sel_y_plus = {**sel_y, **coords}
                    target = self.get_target(var_name_x, sel_x_plus, var_name_y, sel_y_plus)

                    aes_kwargs = self.get_aes_kwargs(aes, var_name_x, aes_sel)
                    fun_kwargs = {**aes_kwargs, **kwargs}
                    fun_kwargs["backend"] = self.backend
                    if subset_info:
                        fun_kwargs = {
                            **fun_kwargs,
                            "var_name_x": var_name_x,
                            "sel_x": sel_x,
                            "isel_x": isel_x,
                            "var_name_y": var_name_y,
                            "sel_y": sel_y,
                            "isel_y": isel_y,
                        }

                    aux_artist = fun(da_x, da_y, target=target, **fun_kwargs)
                    if store_artist:
                        if np.size(aux_artist) == 1:
                            aux_artist = np.squeeze(aux_artist)
                        self.store_in_artist_da(
                            aux_artist,
                            fun_label,
                            var_name_x=var_name_x,
                            sel_x=sel_x,
                            var_name_y=var_name_y,
                            sel_y=sel_y,
                        )
