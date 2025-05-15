# pylint: disable=too-many-lines, too-many-public-methods
"""Plot matrix class."""
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import rcParams, xarray_sel_iter

from arviz_plots.plot_collection import PlotCollection, concat_model_dict


def subset_matrix_da(
    da, var_name_x, selection_x, var_name_y=None, selection_y=None, return_dataarray=False
):
    """Get a subset of a matrix-like DataArray.

    This function assumes that `da` has two dimensions with the same base coordinate names each,
    the only difference being the coords of one dimension have ``_x`` as suffix whereas the
    coords of the other have ``_y`` as suffix. Consequently, the two dimensions are referred to
    as the x dimension and the y dimension.

    Parameters
    ----------
    var_name_x : hashable
        Variable name along the y dimension.
    selection_x : mapping
        Mapping defining the coordinate subset along the x dimension.
    var_name_y : hashable, optional
        Variable name along the y dimension. If not provided it is assumed to be the same
        as `var_name_x`
    selection_y : mapping, optional
        Mapping defining the coordinate subset along the y dimension.
        If not provided it is assumed to be the same as `selection_x`
    return_dataarray : bool, default False
        If true, return the subset of a dataarray. Otherwise, the output is a numpy array
        for multidimensional subsets and the stored object itself if the output is a scalar.
    """
    if (var_name_y is None) and (selection_y is None):
        var_name_y = var_name_x
        selection_y = selection_x
    if any(elem is None for elem in (var_name_x, var_name_y, selection_x, selection_y)):
        raise ValueError("Invalid values for subset arguments")
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

    See Also
    --------
    arviz_plots.PlotCollection : Unidimensional facetting manager
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
            See :meth:`~arviz_plots.PlotMatrix.generate_aes_dt` for more details.
        backend : str, optional
            Plotting backend. It will be stored and passed down to the plotting
            functions when using methods like :meth:`~arviz_plots.PlotMatrix.map`.
        **kwargs : mapping, optional
            Dictionary with :term:`aesthetics` as keys and as values a list
            of the values that should be taken by that aesthetic.
        """
        self._data = concat_model_dict(data)
        self._facet_dims = facet_dims

        if backend is None:
            backend = rcParams["plot.backend"]
        self.backend = backend

        if plot_grid_kws is None:
            plot_grid_kws = {}

        super().__init__(
            data=self._data,
            viz_dt=self._generate_viz_dt(**plot_grid_kws),
            aes=aes,
            backend=backend,
            **kwargs,
        )

    @property
    def facet_dims(self):
        """Facetting dimensions."""
        return set(dim for dim in self._facet_dims if dim != "__variable__")

    def _generate_viz_dt(self, **plot_grid_kws):
        """Generate ``.viz`` DataTree."""
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
        return xr.DataTree(
            xr.Dataset(
                {
                    "figure": np.array(fig, dtype=object),
                    "plot": (("row_index", "col_index"), ax_ary),
                },
                coords=coords,
            )
        )

    def get_target(self, var_name, selection, var_name_y=None, selection_y=None):
        """Get the target that corresponds to the given variable and selection.

        Parameters
        ----------
        var_name : hashable
            Variable name corresponding to the x dimension.
        selection : mapping
            Mapping with with coordinate subset along the x dimension.
        var_name_y : hashable, optional
            Variable name corresponding to the y dimension.
            If not provided it will be assumed as being `var_name`
        selection_y : mapping, optional
            Mapping with with coordinate subset along the y dimension.
            If not provided it will be assumed as being `selection`
        """
        return subset_matrix_da(
            self.viz["plot"],
            var_name_x=var_name,
            selection_x=selection,
            var_name_y=var_name_y,
            selection_y=selection_y,
        )

    def allocate_artist(
        self, fun_label, data, all_loop_dims, dim_to_idx=None, artist_dims=None, ignore_aes=None
    ):
        """Allocate an artist in the ``viz`` DataTree."""
        if artist_dims is None:
            artist_dims = {}
        if dim_to_idx:
            raise ValueError("dim_to_idx not supported yet for PlotMatrix")
        attrs = None
        if ignore_aes is not None:
            attrs = {"ignore_aes": ignore_aes}
        matrix_sizes = self.viz["plot"].sizes
        aes_dims = [dim for dim in data.dims if dim not in self.facet_dims and dim in all_loop_dims]
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

    def store_in_artist_da(self, aux_artist, fun_label, var_name, sel, var_name_y=None, sel_y=None):
        """Store artist object or array into its preallocated DataArray within ``viz``.

        Parameters
        ----------
        aux_artist
            The plotting backend class representing an artist to be stored or
            an array-like of such objects.
        fun_label : hashable
            The identifier of the artist within the ``PlotMatrix``.
            It should be one of the values for which
            :meth:`~arviz_plots.PlotMatrix.allocate_artist` has already been called.
        var_name : hashable
            Variable name corresponding to the x dimension.
        sel : mapping
            Mapping with with coordinate subset along the x dimension.
        var_name_y : hashable, optional
            Variable name corresponding to the y dimension.
            If not provided it will be assumed as being `var_name`
        sel_y : mapping, optional
            Mapping with with coordinate subset along the y dimension.
            If not provided it will be assumed as being `sel`
        """
        plot_da = subset_matrix_da(
            self.viz["plot"],
            var_name_x=var_name,
            selection_x=sel,
            var_name_y=var_name_y,
            selection_y=sel_y,
            return_dataarray=True,
        )
        self._viz_dt[fun_label].loc[
            {"row_index": plot_da["row_index"], "col_index": plot_da["col_index"]}
        ] = aux_artist

    def map_upper(self, *args, **kwargs):
        """Call :meth:`~arviz_plots.PlotMatrix.map_triangle` with ``triangle="upper"``."""
        self.map_triangle(*args, triangle="upper", **kwargs)

    def map_lower(self, *args, **kwargs):
        """Call :meth:`~arviz_plots.PlotMatrix.map_triangle` with ``triangle="lower"``."""
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
        """Apply the given plotting function to all plots with the corresponding aesthetics.

        Parameters
        ----------
        fun : callable
            Function with signature ``fun(da_x, da_y, target, **fun_kwargs)`` which
            should be called for all couples of data pairs (each couple encoded in a :term:`plot`)
            and corresponding :term:`aesthetic`.
            The object returned by `fun` is assumed to be an scalar unless
            `artist_dims` are provided. There is also the option of adding extra
            keyword arguments with the `subset_info` flag.
        fun_label : str, optional
            Function identifier. It will be used as variable name to store the object
            returned by `fun`. Defaults to ``fun.__name__``.
        data : Dataset, optional
            Data to be subsetted into pair elements then loop to cover all couple combinations.
            Defaults to the data used to initalize the ``PlotMatrix``.
        loop_data : Dataset or str
            TODO: see if it works and if we want to keep it.
        coords : mapping, optional
            Dictionary of {coordinate names : coordinate values} that should
            be used to subset the aes, data and viz objects before any faceting
            or aesthetics mapping is applied.
        ignore_aes : set, optional
            Set of aesthetics present in ``aes`` that should be ignore for this
            ``map`` call.
        subset_info : boolean, default False
            Add the subset info from :func:`arviz_base.xarray_sel_iter`
            for the ``da_x``+``da_y`` couple to the keyword arguments passed to `fun`.
            If true, then `fun` must accept the keyword arguments ``var_name_x``,
            ``sel_x``, ``isel_x``, ``var_name_y``, ``sel_y`` and ``isel_y``.
            Moreover, if those were to be keys present in `**kwargs` their
            values in `**kwargs` would be ignored.
        store_artist : boolean, default True
        artist_dims : mapping of {hashable : int}, optional
            Dictionary of sizes for proper allocation and storage when using
            ``map`` with functions that return an array of :term:`artist`.
        **kwargs : mapping, optional
            Extra keyword arguments to be passed to `fun`.

        See Also
        --------
        arviz_plots.PlotMatrix.map
        """
        if triangle not in {"lower", "upper", "both"}:
            raise ValueError(
                "Invalid value for `triangle` options are 'lower', 'upper' or 'both' "
                f"but got {triangle}"
            )
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

        facet_dims = self.facet_dims
        aes, all_loop_dims = self.update_aes(ignore_aes, coords)
        aes_dims = [dim for dim in all_loop_dims if dim not in facet_dims]
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
                loop_data, skip_dims={dim for dim in loop_data.dims if dim not in facet_dims}
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
                            var_name_x,
                            sel_x,
                            var_name_y=var_name_y,
                            sel_y=sel_y,
                        )

    def map(
        self,
        fun,
        fun_label=None,
        *,
        data=None,
        loop_data=None,
        coords=None,
        ignore_aes=frozenset(),
        subset_info=False,
        store_artist=True,
        artist_dims=None,
        **kwargs,
    ):
        """Apply the given plotting function along the diagonal with the corresponding aesthetics.

        Parameters
        ----------
        fun : callable
            Function with signature ``fun(da, target, **fun_kwargs)`` which should
            be applied for all combinations of :term:`plot` and :term:`aesthetic`.
            The object returned by `fun` is assumed to be a scalar unless
            `artist_dims` are provided. There is also the option of adding
            extra required keyword arguments with the `subset_info` flag.
        fun_label : str, optional
            Variable name with which to store the object returned by `fun`.
            Defaults to ``fun.__name__``.
        data : Dataset, optional
            Data to be subsetted at each iteration and to pass to `fun` as first positional
            argument. Defaults to the data used to initialize the ``PlotMatrix``.
        loop_data : Dataset or str, optional
            Data which will be used to loop over and generate the information used to subset
            `data`. It also accepts the value "plots" as a way to indicate `fun` should be
            applied exactly once per :term:`plot`. Defaults to the value of `data`.
        coords : mapping, optional
            Dictionary of {coordinate names : coordinate values} that should
            be used to subset the aes, data and viz objects before any faceting
            or aesthetics mapping is applied.
        ignore_aes : set, optional
            Set of aesthetics present in ``aes`` that should be ignore for this
            ``map`` call.
        subset_info : boolean, default False
            Add the subset info from :func:`arviz_base.xarray_sel_iter` to
            the keyword arguments passed to `fun`. If true, then `fun` must
            accept the keyword arguments ``var_name``, ``sel`` and ``isel``.
            Moreover, if those were to be keys present in `**kwargs` their
            values in `**kwargs` would be ignored.
        store_artist : boolean, default True
        artist_dims : mapping of {hashable : int}, optional
            Dictionary of sizes for proper allocation and storage when using
            ``map`` with functions that return an array of :term:`artist`.
        **kwargs : mapping, optional
            Keyword arguments passed as is to `fun`. Values within `**kwargs`
            with :class:`~xarray.DataArray` of :class:`~xarray.Dataset` type
            will be subsetted on the current selection (if possible) before calling `fun`.
            Slicing with dims and coords is applied to the relevant subset present in the
            xarray object so dimensions with mapped asethetics not being present is not an issue.
            However, using Datasets that don't contain all the variable names in `data`
            will raise an error.

        See Also
        --------
        arviz_plots.PlotMatrix.map_triangle
        """
        super().map(
            fun=fun,
            fun_label=fun_label,
            data=data,
            loop_data=loop_data,
            coords=coords,
            ignore_aes=ignore_aes,
            subset_info=subset_info,
            store_artist=store_artist,
            artist_dims=artist_dims,
            **kwargs,
        )

    @property
    def viz(self):
        """Information about the visual elements in the plot as a DataTree.

        The DataTree only has variables in the root group.
        With all variables having the same dimensions: ``(row_index, col_index)``.
        The information about facetting is encoded in the coordinate values;
        ``row_index`` has all relevant coordinates to indicate the subset with ``_y`` suffix,
        ``col_index`` has coordinates with the ``_x`` suffix.
        The `viz` DataTree always contains the following variables:

        * ``figure`` (always on the home group) -> Scalar object containing the highest level
          plotting structure. i.e. the matplotlib figure or the bokeh layout
        * ``plot`` -> :term:`Plot` objects in this :term:`figure`.
          Generally, these are the target where :term:`artists <artist>` are added,
          although it is possible to have artists targetting the figure itself.

        Plus all the artists that have been added to the plot and stored.
        See :meth:`arviz_plots.PlotMatrix.map` and :meth:`arviz_plots.PlotMatrix.map_triangle`
        for more details.
        """
        if self.coords is None:
            return self._viz_dt
        raise ValueError("viz attribute can't be accessed with coords set")
