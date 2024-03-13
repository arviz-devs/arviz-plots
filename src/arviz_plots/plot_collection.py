"""Plot collection classes."""
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.sel_utils import xarray_sel_iter
from datatree import DataTree


def sel_subset(sel, present_dims):
    """Subset a dictionary of dim: coord values.

    The returned dictionary contains only the keys that
    are present to ensure we can use the output of this function
    to index correctly using ``.sel``.
    """
    return {key: value for key, value in sel.items() if key in present_dims}


def subset_ds(ds, var_name, sel):
    """Subset a dataset in a non-idempotent way.

    Get a subset indicated by `sel` of the variable in the Dataset indicated by `var_names`
    and return a scalar or a numpy array. This helps with getting the proper matplotlib
    axes or bokeh figure, converting the DataArrays we get from ``.sel`` to arrays to ensure
    compatibility with all plotting backends... without having to add ``.item()`` or ``.value``
    constantly in the code. It also calls :func:`sel_subset` to ensure ``.sel`` doesn't error.
    The variable name indicated by `var_name` needs to exist though.

    Parameters
    ----------
    ds : Dataset
    var_name : hashable
    sel : mapping
    """
    subset_dict = sel_subset(sel, ds[var_name].dims)
    if subset_dict:
        out = ds[var_name].sel(subset_dict)
    else:
        out = ds[var_name]
    if out.size == 1:
        return out.item()
    return out.values


def _process_facet_dims(data, facet_dims):
    """Process facetting dimensions.

    It takes into account the ``__variable__`` "special dimension name" and helps find out
    how many plots are needed.
    """
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


class PlotCollection:
    """Low level base class for plotting with xarray Datasets.

    This class instatiates a chart with multiple plots in it and provides methods to loop
    over these plots and the provided data syncing each plot and data subset to
    user given aesthetics.

    Attributes
    ----------
    viz : DataTree
        DataTree containing all the visual elements in the plot. If relevant, the variable
        names in the input Dataset are set as groups, otherwise everything is stored in the
        home group. The `viz` DataTree always contains the following variables:

        * ``chart`` (always on the home group): Scalar object containing the highest level
          plotting structure. i.e. the matplotlib figure or the bokeh layout
        * ``plot``: :term:`Plot` objects in this :term:`chart`.
          Generally, these are the target where :term:`artists <artist>` are added,
          although it is possible to have artists targetting the chart itself.
        * ``row``: Integer row indicator
        * ``col``: Integer column indicator

        Plus all the artists that have been added to the plot and stored.
        See :meth:`arviz_plots.PlotCollection.map` for more details.
    aes : DataTree
        DataTree containing the :term:`aesthetic mapping` information.
        A subset of the input dataset ``ds[var_name].sel(**kwargs)``
        is associated the aesthetics in ``aes[var_name].sel(**kwargs)``.
        Note that here `aes` is a DataTree so ``aes[var_name]`` is a Dataset.
        There can be as many aesthetic mappings as desired,
        and they can map to any dimensions *independently from one another*
        and also independently between variables (even if not recommended).
    """

    def __init__(self, data, viz_dt, aes_dt=None, aes=None, backend=None, **kwargs):
        """Initialize a PlotCollection.

        It is not recommeded to initialize ``PlotCollection`` objects directly.
        Use its classmethods :meth:`~arviz_plots.PlotCollection.wrap` and
        :meth:`~arviz_plots.PlotCollection.grid` instead.

        Parameters
        ----------
        data : Dataset
            The data from which `viz_dt` was generated and
            from which to generate the aesthetic mappings.
        viz_dt : DataTree
            DataTree object with which to populate the ``viz`` attribute.
        aes_dt : DataTree, optional
            DataTree object with which to populate the ``aes`` attribute.
            If given, the `aes` argument and all `**kwargs` are ignored.
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

        See Also
        --------
        arviz_plots.PlotCollection.grid, arviz_plots.PlotCollection.wrap
        """
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
        """Dataset to be used as data for plotting."""
        return self._data

    @data.setter
    def data(self, value):
        # might want/be possible to make some checks on the data before setting it
        self._data = value

    @property
    def aes_set(self):
        """Return all aesthetics with a mapping defined as a set."""
        return set(self._aes.keys())

    def show(self):
        """Call the backend function to show this :term:`chart`."""
        if "chart" not in self.viz:
            raise ValueError("No plot found to be shown")
        plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")
        plot_bknd.show(self.viz["chart"].item())

    def generate_aes_dt(self, aes, **kwargs):
        """Generate the aesthetic mappings.

        Parameters
        ----------
        aes : mapping of {str : list of hashable}, optional
            Dictionary with :term:`aesthetics` as keys and as values a list
            of the dimensions it should be mapped to.
            See :meth:`~arviz_plots.PlotCollection.generate_aes_dt` for more details.
        **kwargs : mapping, optional
            Dictionary with :term:`aesthetics` as keys and as values a list
            of the values that should be taken by that aesthetic.

        Returns
        -------
        aes_dt : DataTree
            DataTree object to be stored as ``.aes`` attribute of the PlotCollection.

        Examples
        --------
        Initialize a `PlotCollection` with the rugby dataset as data.
        Facetting and aesthetics mapping are independent. Thus, as
        we are limiting ourselves to the use of this method, we can
        provide an empty DataTree as ``viz_dt``.

        .. jupyter-execute::

            from datatree import DataTree
            from arviz_base import load_arviz_data
            from arviz_plots import PlotCollection
            from arviz_base.datasets import REMOTE_DATASETS, RemoteFileMetadata
            # TODO: remove this monkeypatching once the arviz_example_data repo has been updated
            REMOTE_DATASETS.update({
                "rugby_field": RemoteFileMetadata(
                    name="rugby_field",
                    filename="rugby_field.nc",
                    url="http://figshare.com/ndownloader/files/44667112",
                    checksum="53a99da7ac40d82cd01bb0b089263b9633ee016f975700e941b4c6ea289a1fb0",
                    description="Variant of the rugby model."
                )
            })
            idata = load_arviz_data("rugby_field")
            pc = PlotCollection(idata.posterior, DataTree())
            pc.generate_aes_dt(
                aes={
                    "color": ["team"],
                    "y": ["field", "team"],
                    "linestyle": ["field"],
                },
                color=[f"C{i}" for i in range(6)],
                y=list(range(12)),
                linestyle=["-", ":"],
            )
            pc.aes

        The generated `aes_dt` has one group per variable in the posterior group
        in the provided data. Each group in the `aes_dt` DataTree is a Dataset
        with the aesthetics that apply to that variable and the required shape
        for all values that aesthetic needs to take.

        Thus, when we subset the data for plotting with
        ``ds[var_name].sel(**kwargs)`` we can get its aesthetics with
        ``aes_dt[var_name].sel(**kwargs)``.

        Notes
        -----
        All values for the provided aesthetics to take need to be given
        manually through the `**kwargs` arguments. To allow for any arbitrary
        argument of the plotting functions called later and to ensure support
        for all backends manual entry of values is a must.

        In the future, it may be possible to skip the `**kwargs` corresponding
        to aesthetics that are part of the common interface in :mod:`arviz_plots.backend`,
        but it will always be possible to set their value manually.
        """
        if aes is None:
            aes = {}
        self._aes = aes
        self._kwargs = kwargs
        self.aes = DataTree()
        for var_name, da in self.data.items():
            ds = xr.Dataset()
            for aes_key, dims in aes.items():
                aes_vals = kwargs.get(aes_key, [None])
                aes_dims = [dim for dim in dims if dim in da.dims]
                aes_raw_shape = [da.sizes[dim] for dim in aes_dims]
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
                    dims=aes_dims,
                    coords={dim: da.coords[dim] for dim in dims if dim in da.coords},
                )
            DataTree(name=var_name, parent=self.aes, data=ds)

    @property
    def base_loop_dims(self):
        """Dimensions over which one should always loop over when using this PlotCollection."""
        if "plot" in self.viz.data_vars:
            return set(self.viz["plot"].dims)
        return set(dim for da in self.viz.children.values() for dim in da["plot"].dims)

    def get_viz(self, var_name):
        """Get the ``viz`` Dataset that corresponds to the provided variable."""
        return self.viz if "plot" in self.viz.data_vars else self.viz[var_name]

    @classmethod
    def wrap(
        cls,
        data,
        cols=None,
        col_wrap=4,
        backend=None,
        plot_grid_kws=None,
        **kwargs,
    ):
        """Instatiate a PlotCollection and generate a plot grid iterating over subsets and wrapping.

        Parameters
        ----------
        data : Dataset or dict of {str: Dataset}
            If `data` is a dictionary, the Datasets stored as its values will be concatenated,
            creating a new dimension called ``model``.
        cols : iterable of hashable, optional
            Dimensions of the dataset for which different coordinate values
            should have different :term:`plots <plot>`. A special dimension
            called ``__variable__`` is also available, to indicate that
            each variable of the input Dataset should have their own plot;
            it can also be combined with other dimensions.
        col_wrap : int, default 4
            Number of columns in the generated grid. If more than `col_wrap`
            plots are needed from :term:`facetting` according to `cols`,
            new rows are created.
        backend : str, optional
            Plotting backend.
        plot_grid_kws : mapping, optional
            Passed to ``create_axis_grid`` of the chosen plotting backend.
        **kwargs : mapping, optional
            Passed as is to the initializer of ``PlotCollection``. That is,
            used for ``aes`` and ``**kwargs`` arguments.
            See :meth:`~arviz_plots.PlotCollection.generate_aes_dt` for more
            details about these arguments.

        See Also
        --------
        arviz_plots.PlotCollection.grid
        """
        if cols is None:
            cols = []
        if plot_grid_kws is None:
            plot_grid_kws = {}
        if backend is None:
            backend = rcParams["plot.backend"]
        if isinstance(data, dict):
            data = xr.concat(data.values(), dim="model").assign_coords(model=list(data))

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
                    },
                    coords={dim: da[dim] for dim in dims},
                )
        viz_dt = DataTree.from_dict(viz_dict)
        return cls(data, viz_dt, backend=backend, **kwargs)

    @classmethod
    def grid(
        cls,
        data,
        cols=None,
        rows=None,
        backend=None,
        plot_grid_kws=None,
        **kwargs,
    ):
        """Instatiate a PlotCollection and generate a plot grid iterating over rows and columns.

        Parameters
        ----------
        data : Dataset or dict of {str: Dataset}
            If `data` is a dictionary, the Datasets stored as its values will be concatenated,
            creating a new dimension called ``model``.
        cols, rows : iterable of hashable, optional
            Dimensions of the dataset for which different coordinate values
            should have different :term:`plots <plot>`. A special dimension
            called ``__variable__`` is also available, to indicate that
            each variable of the input Dataset should have their own plot;
            it can also be combined with other dimensions.

            The generated grid will have as many plots as unique combinations
            of values within `cols` and `rows`.
        backend : str, optional
            Plotting backend.
        plot_grid_kws : mapping, optional
            Passed to ``create_axis_grid`` of the chosen plotting backend.
        **kwargs : mapping, optional
            Passed as is to the initializer of ``PlotCollection``. That is,
            used for ``aes`` and ``**kwargs`` arguments.
            See :meth:`~arviz_plots.PlotCollection.generate_aes_dt` for more
            details about these arguments.

        See Also
        --------
        arviz_plots.PlotCollection.wrap
        """
        if cols is None:
            cols = []
        if rows is None:
            rows = []
        if plot_grid_kws is None:
            plot_grid_kws = {}
        if backend is None:
            backend = rcParams["plot.backend"]
        repeated_dims = [col for col in cols if col in rows]
        if repeated_dims:
            raise ValueError("The same dimension can't be used for both cols and rows.")
        if isinstance(data, dict):
            data = xr.concat(data.values(), dim="model").assign_coords(model=list(data))

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
                    },
                    coords={dim: da[dim] for dim in dims},
                )
        viz_dt = DataTree.from_dict(viz_dict)
        return cls(data, viz_dt, backend=backend, **kwargs)

    def update_aes(self, ignore_aes=frozenset(), coords=None):
        """Update list of aesthetics after indicating ignores and extra subsets."""
        if coords is None:
            coords = {}
        aes = [aes_key for aes_key in self._aes.keys() if aes_key not in ignore_aes]
        aes_dims = [dim for aes_key in aes for dim in self._aes[aes_key]]
        all_loop_dims = self.base_loop_dims.union(aes_dims).difference(coords.keys())
        return aes, all_loop_dims

    def allocate_artist(self, fun_label, data, all_loop_dims, artist_dims=None):
        """Allocate an artist in the ``viz`` DataTree."""
        if artist_dims is None:
            artist_dims = {}
        for var_name, da in data.items():
            if var_name not in self.viz.children:
                DataTree(name=var_name, parent=self.viz)
            inherited_dims = [dim for dim in da.dims if dim in all_loop_dims]
            artist_shape = [da.sizes[dim] for dim in inherited_dims] + list(artist_dims.values())
            all_artist_dims = inherited_dims + list(artist_dims.keys())

            self.viz[var_name][fun_label] = xr.DataArray(
                np.empty(artist_shape, dtype=object),
                dims=all_artist_dims,
                coords={dim: data[dim] for dim in inherited_dims},
            )

    def get_target(self, var_name, selection):
        """Get the target that corresponds to the given variable and selection."""
        return subset_ds(self.get_viz(var_name), "plot", selection)

    def get_aes_kwargs(self, aes, var_name, selection):
        """Get the aesthetic mappings for the given variable and selection as a dictionary."""
        aes_kwargs = {}
        if var_name not in self.aes.data_vars:
            return aes_kwargs
        for aes_key in aes:
            aes_kwargs[aes_key] = subset_ds(self.aes[var_name], aes_key, selection)
        return aes_kwargs

    def plot_iterator(self, ignore_aes=frozenset(), coords=None):
        """Build a generator to loop over all plots in the PlotCollection."""
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
        loop_data=None,
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
            Function with signature ``fun(da, target, **fun_kwargs)`` which should
            be applied for all combinations of :term:`plot` and :term:`aesthetic`.
            The object returned by `fun` is assumed to be a scalar unless
            `artist_dims` are provided. There is also the option of adding
            extra required keyword arguments with the `subset_info` flag.
        fun_label : str, optional
            Variable name with which to store the object returned by `fun`.
            Defaults to ``fun.__name__``.
        data : Dataset, optional
            Data to be subsetted at each iteration and to pass to `fun` as first argument.
            Defaults to the data used to initialize the ``PlotCollection``.
        loop_data : Dataset or str, optional
            Data which will be used to loop over and generate the information used to subset
            `data`. It also accepts the value "plots" as a way to indicate `fun` should be
            applied exactly once per :term:`plot`. Defaults to the value of `data`.
        coords : mapping, optional
            Dictionary of {coordinate names : coordinate values} that should
            be used to subset the aes, data and viz objects before any facetting
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
            Keyword arguments passed as is to `fun`.

        See Also
        --------
        map_over_plots
        """
        if coords is None:
            coords = {}
        if self.aes is None:
            self.generate_aes_dt(self._aes, **self._kwargs)
        if fun_label is None:
            fun_label = fun.__name__

        data = self.data if data is None else data
        if isinstance(loop_data, str) and loop_data == "plots":
            if "plot" in self.viz.data_vars:
                loop_data = self.viz[["plot"]]
            else:
                loop_data = xr.Dataset(
                    {var_name: ds["plot"] for var_name, ds in self.viz.children.items()}
                )
        loop_data = data if loop_data is None else loop_data

        aes, all_loop_dims = self.update_aes(ignore_aes, coords)
        plotters = xarray_sel_iter(
            loop_data, skip_dims={dim for dim in loop_data.dims if dim not in all_loop_dims}
        )
        if store_artist:
            self.allocate_artist(
                fun_label=fun_label,
                data=loop_data,
                all_loop_dims=all_loop_dims,
                artist_dims=artist_dims,
            )

        for var_name, sel, isel in plotters:
            da = data[var_name].sel(sel)
            if np.all(np.isnan(da)):
                continue
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

    def add_legend(self, dim, var_name=None, aes=None, artist_kwargs=None, title=None, **kwargs):
        """Add a legend for the given artist/aesthetic to the plot.

        Warnings
        --------
        This method is still in early stages of experimentation and anything beyond
        the basic usage ``add_legend("dim_name")`` will probably change in breaking ways.

        Parameters
        ----------
        dim : hashable
            Dimension for which to generate the legend. It should have at least
            one :term:`aesthetic mapped <aesthetic mapping>` to it.
        var_name : hashable, optional
            Variable name for which to generate the legend. Unless the ``aes``
            DataTree has been modified manually, the legend will be independent
            of the variable chosen. Defaults to the first variable with `dim` as
            dimension.
        aes : str or iterable of str, optional
            Specific aesthetics to take into account when generating the legend.
            They should all be mapped to `dim`.
        artist_kwargs : mapping, optional
            Keyword arguments passed to the backend artist function used to
            generate the miniatures in the legend.
        title : str, optional
            Legend title. Defaults to `dim`.
        **kwargs : mapping, optional
            Keyword arguments passed to the backend function that generates the legend.

        Returns
        -------
        legend : object
            The corresponding legend object for the backend of the ``PlotCollection``.
        """
        if title is None:
            title = dim
        if var_name is None:
            var_name = [name for name, ds in self.aes.children.items() if dim in ds.dims][0]
        aes_ds = self.aes[var_name].to_dataset()
        if dim not in aes_ds.dims:
            raise ValueError(
                f"Legend can't be generated. Found no aesthetics mapped to dimension {dim}"
            )
        aes_ds = aes_ds.drop_dims([d for d in aes_ds.dims if d != dim])
        if aes is None:
            dropped_vars = ["x", "y"] + [name for name, da in aes_ds.items() if dim not in da.dims]
            aes_ds = aes_ds.drop_vars(dropped_vars, errors="ignore")
        else:
            if isinstance(aes, str):
                aes = [aes]
            aes_ds = aes_ds[aes]
        label_list = aes_ds[dim].values
        kwarg_list = [
            {k: v.item() for k, v in aes_ds.sel({dim: coord}).items()} for coord in label_list
        ]
        plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")
        return plot_bknd.legend(
            self.viz["chart"].item(),
            kwarg_list,
            label_list,
            title=title,
            artist_kwargs=artist_kwargs,
            **kwargs,
        )
