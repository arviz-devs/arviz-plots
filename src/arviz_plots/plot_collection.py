# pylint: disable=too-many-lines, too-many-public-methods
"""Plot collection class."""
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import xarray as xr
from arviz_base import rcParams, xarray_sel_iter
from arviz_base.labels import BaseLabeller


def backend_from_object(obj, return_module=True):
    """Get the backend string or module that corresponds to a given object.

    Parameters
    ----------
    obj
        The object to get its corresponding backend for.
    return_module : bool, default True
        Return the module from ``arviz_plots.backend` after importing it

    Returns
    -------
    backend : module or str
    """
    # cover none backend first, the figure object is a dictionary,
    # and the plot objects are lists
    if isinstance(obj, list | dict):
        backend = "none"
    else:
        lib, *_, leaf, _ = obj.__module__.split(".")
        # for plotly, the target will actually be an arviz_plots.backend.plotly.core.PlotlyPlot
        if lib == "arviz_plots":
            backend = leaf
        else:
            backend = lib
    if backend == "xarray":
        raise ValueError(
            "Attempting to get plotting backend from a non-scalar object. "
            "Review faceting and aesthetic mapping strategies. Conflicting xarray object:\n"
            f"{obj}"
        )
    if return_module:
        return import_module(f"arviz_plots.backend.{backend}")
    return backend


def concat_model_dict(data):
    """Merge multiple Datasets into a single one along a new model dimension."""
    if isinstance(data, dict):
        ds_list = data.values()
        if not all(isinstance(ds, xr.Dataset) for ds in ds_list):
            raise TypeError("Provided data must be a Dataset or dictionary of Datasets")
        data = xr.concat(ds_list, dim="model").assign_coords(model=list(data))
    return data


def sel_subset(sel, ds_da):
    """Subset a dictionary of dim: coord values.

    The returned dictionary contains only the keys that
    are present to ensure we can use the output of this function
    to index correctly using ``.sel``.

    Preference is given to indexers with the same name as the dimension,
    but
    """
    dim_subset = {key: value for key, value in sel.items() if key in ds_da.dims}
    dims_with_coords = list(dim_subset)
    for key in sel:
        if key in dim_subset:
            continue
        if key in ds_da.coords:
            da_indexer = ds_da[key]
            if da_indexer.ndim == 1 and da_indexer.dims[0] not in dims_with_coords:
                dim_subset[key] = sel[key]
                dims_with_coords.append(da_indexer.dims[0])
    return dim_subset


def subset_ds(ds, var_name, sel):
    """Subset a dataset in a potentially non-idempotent way.

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
    ds = ds[var_name]
    if isinstance(ds, xr.DataTree):
        ds = ds.dataset
    subset_dict = sel_subset(sel, ds)
    if subset_dict:
        for key in subset_dict:
            if key not in ds.dims and key not in ds.xindexes:
                ds = ds.set_xindex(key)
        out = ds.sel(subset_dict)
    else:
        out = ds
    if out.size == 1:
        return out.item()
    return out.values


def try_da_subset(da, sel):
    """Try subsetting a dataarray with `.sel`.

    There are 3 possible cases:

    * None of the keys in `sel` are dimensions in `da` -> `da` is returned as is
    * Some (or all) of the keys in `sel` are dimensions in `da`:

      - `.sel` on the subset of dimensions present works -> return `da` subset
      - `.sel` raises a KeyError -> return ``None``
    """
    subset_dict = sel_subset(sel, da)
    if subset_dict:
        for key in subset_dict:
            if key not in da.xindexes:
                da = da.set_xindex(key)
        try:
            da = da.sel(subset_dict)
        except KeyError:
            return None
    return da


def process_kwargs_subset(value, var_name, sel):
    """Process kwargs to subset xarray objects if possible.

    Anything not a Dataset or DataArray is returned as is.
    """
    if isinstance(value, xr.Dataset):
        if var_name not in value.data_vars:
            subset_dict = sel_subset(sel, value)
            if subset_dict:
                try:
                    ds = value.sel(subset_dict)
                except KeyError:
                    return None
                return ds
            return value
        value = value[var_name]
    if isinstance(value, xr.DataArray):
        return try_da_subset(value, sel)
    return value


def process_facet_dims(data, facet_dims):
    """Process faceting dimensions.

    It takes into account the ``__variable__`` "special dimension name" and helps find out
    how many plots are needed.
    """
    if not facet_dims:
        return 1, {}
    facets_per_var = {}
    if "__variable__" in facet_dims:
        for var_name, da in data.items():
            lenghts = [
                len(np.unique(da[dim]))
                for dim in facet_dims
                if dim in set(da.dims).union(da.coords)
            ]
            facets_per_var[var_name] = np.prod(lenghts) if lenghts else 1
        n_facets = np.sum(list(facets_per_var.values()))
    else:
        missing_dims = {
            var_name: [dim for dim in facet_dims if dim not in set(da.dims).union(da.coords)]
            for var_name, da in data.items()
        }
        missing_dims = {k: v for k, v in missing_dims.items() if v}
        if any(missing_dims.values()):
            raise ValueError(
                "All variables must have all faceting dimensions, but found the following "
                f"dims to be missing in these variables: {missing_dims}"
            )
        n_facets = np.prod([len(np.unique(data[dim])) for dim in facet_dims])
    return n_facets, facets_per_var


def leaf_dataset(dt, leaf_name):
    """Get leaf nodes named `leaf_name` from `dt`.

    Parameters
    ----------
    dt : DataTree
    leaf_name : hashable

    Returns
    -------
    Dataset
    """
    return xr.Dataset({var_name: values[leaf_name] for var_name, values in dt.children.items()})


class PlotCollection:
    """Low level base class for plotting with xarray Datasets.

    This class instantiates a figure with multiple plots in it and provides methods to loop
    over these plots and the provided data syncing each plot and data subset to
    user given aesthetics.

    Attributes
    ----------
    viz : DataTree
    aes : DataTree

    See Also
    --------
    arviz_plots.PlotMatrix : Pairwise facetting manager
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
        self._coords = None
        self._viz_dt = viz_dt

        if backend is not None:
            self.backend = backend
        elif "figure" in viz_dt:
            self.backend = backend_from_object(self.viz_dt["figure"].item(), return_module=False)

        if aes_dt is None:
            if aes is None:
                aes = {}
            self._aes_dt = self.generate_aes_dt(aes, data, **kwargs)
        else:
            self._aes_dt = aes_dt

    @property
    def aes(self):
        """Information about :term:`aesthetic mapping` as a DataTree.

        For aesthetics where the variable is used to encode information
        (that is, "__variable__" was used, a subset of the input dataset
        ``ds[var_name].sel(**kwargs)`` is associated the aesthetics in
        ``aes[aes_key][var_name].sel(**kwargs)``.

        For aesthetics mappping that only use dimensions for mapping the dataset
        will have a variable "mapping" with shape inherited from the mapped dimensions
        in the original data, and might also have a "neutral_element" scalar
        variable.

        The docstring for :meth:`arviz_plots.PlotCollection.generate_aes_dt`
        has examples and covers the "neutral element" concept in more detail.

        See Also
        --------
        .PlotCollection.generate_aes_dt
        .PlotCollection.get_aes_kwargs
        """
        if self.coords is None:
            return self._aes_dt
        return xr.DataTree.from_dict(
            {
                group: ds.to_dataset().sel(sel_subset(self.coords, ds))
                for group, ds in self._aes_dt.children.items()
            }
        )

    @aes.setter
    def aes(self, value):
        if self.coords is not None:
            raise ValueError("Can't modify `aes` DataTree while `coords` is set")
        self._aes_dt = value

    @property
    def viz(self):
        """Information about the visual elements in the plot as a DataTree.

        Plot elements like :term:`visuals`, :term:`plots` and the :term:`figure`
        are stored at the top level, if possible directly as DataArrays,
        otherwise as groups whose variables are variable names in the input
        Dataset.
        The `viz` DataTree always contains the following leaf variables:

        * ``figure`` (always on the home group) -> Scalar object containing the highest level
          plotting structure. i.e. the matplotlib figure or the bokeh layout
        * ``plot`` -> :term:`Plot` objects in this :term:`figure`.
          Generally, these are the target where :term:`visuals <visual>` are added,
          although it is possible to have visuals targetting the figure itself.
        * ``row`` -> Integer row indicator
        * ``col`` -> Integer column indicator

        See :meth:`arviz_plots.PlotCollection.map` for more details.
        """
        if self.coords is None:
            return self._viz_dt
        # TODO: use .loc on DataTree directly (once available), otherwise, changes to
        # .viz aren't stored in the PlotCollection class, same in `aes`
        sliced_viz_dict = {
            group: ds.to_dataset().sel(sel_subset(self.coords, ds))
            for group, ds in self._viz_dt.children.items()
        }
        root_ds = self._viz_dt.to_dataset()
        sliced_viz_dict["/"] = root_ds.sel(sel_subset(self.coords, root_ds))
        return xr.DataTree.from_dict(sliced_viz_dict)

    @viz.setter
    def viz(self, value):
        if self.coords is not None:
            raise ValueError("Can't modify `viz` DataTree while `coords` is set")
        self._viz_dt = value

    @property
    def coords(self):
        """Information about slicing operation to always be applied on the PlotCollection.

        It is similar to the ``coords`` argument in :meth:`~.PlotCollection.map` but
        these coordinates are always taken into account when interfacing with `PlotCollection`,
        even when accessing :attr:`~.PlotCollection.viz` or :attr:`~.PlotCollection.aes`.
        """
        return self._coords

    @coords.setter
    def coords(self, value):
        self._coords = value

    @property
    def data(self):
        """Dataset to be used as data for plotting."""
        return self._data

    @data.setter
    def data(self, value):
        # might want/be possible to make some checks on the data before setting it
        self._data = concat_model_dict(value)

    @property
    def aes_set(self):
        """Return all aesthetics with a mapping defined as a set."""
        return set(self.aes.children)

    def show(self):
        """Call the backend function to show this :term:`figure`."""
        if "figure" not in self.viz:
            raise ValueError("No plot found to be shown")
        plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")
        figure = self.viz["figure"].item()
        if figure is not None:
            plot_bknd.show(figure)
        else:
            self.viz["plot"].item()

    def savefig(self, filename, **kwargs):
        """Call the backend function to save this :term:`figure`.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
        **kwargs
            Passed as is to the respective backend function
        """
        if "figure" not in self.viz:
            raise ValueError("No plot found to be saved")
        plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")
        plot_bknd.savefig(self.viz["figure"].item(), Path(filename), **kwargs)

    def generate_aes_dt(self, aes, data=None, **kwargs):
        """Generate the aesthetic mappings.

        Populate and store the ``DataTree`` attribute ``.aes`` of the ``PlotCollection``.

        Parameters
        ----------
        aes : mapping of {str : list of hashable or False}
            Dictionary with :term:`aesthetics` as keys and as values a list
            of the dimensions it should be mapped to. The pseudo-dimension
            ``__variable__`` is also valid to indicate the variable should be
            part of the aesthetic mapping.

            It can also take ``False`` as value to indicate that no mapping
            should be considered for that aesthetic key.
        data : Dataset, optional
            Data for which to generate the aesthetic mappings.
        **kwargs : mapping, optional
            Dictionary with :term:`aesthetics` as keys and as values a list
            of the values that should be taken by that aesthetic.

        Notes
        -----
        Mappings are applied only when all variables defined in the mapping are found.
        Thus, a mapping for ``["chain", "hierarchy"]`` would be applied if both
        dimensions are present in the variable, otherwise it is completely ignored.

        It can be the case that a mapping is ignored for a specific variable
        because it has none of the dimensions that define the mapping or because
        it doesn't have all of them. In such cases, out of the values in the property
        cycle, the first one is taken out and reserved as *neutral_element*.
        Then, the cycle excluding the first element is used when applying the mapping,
        and the neutral element is used when the mapping can't be applied.

        It is possible to force the inclusion of the neutral element from the
        property value cycle by providing the same value in both the first and second
        positions in the cycle, but this is generally not recommended.

        Examples
        --------
        Initialize a `PlotCollection` with the rugby dataset as data.
        faceting and aesthetics mapping are independent. Thus, as
        we are limiting ourselves to the use of this method, we can
        provide an empty DataTree as ``viz_dt``.

        .. jupyter-execute::

            from arviz_base import load_arviz_data
            from arviz_plots import PlotCollection
            import xarray as xr
            idata = load_arviz_data("rugby_field")
            pc = PlotCollection(idata.posterior, xr.DataTree(), backend="matplotlib")
            aes_dt = pc.generate_aes_dt(
                aes={
                    "color": ["__variable__", "team"],
                    "y": ["field", "team"],
                    "marker": ["field"],
                    "linestyle": ["chain"],
                },
                color=[f"C{i}" for i in range(6)],
                y=list(range(13)),
                linestyle=["-", ":", "--", "-."],
            )

        The generated `aes_dt` has one group per aesthetic. Within each group
        There can be the variables from the Dataset used to initialize the
        PlotCollection or the variables "mapping" and "neutral_element".

        Let's inspect its contents for each aesthetic.
        We'll start with the color which had ``__variable__, team`` as dimensions
        to encode.

        .. jupyter-execute::

            aes_dt["color"]

        In this case, each unique combination of variable and coordinate value of the
        team dimension gets a different color. They only end up being repeated once
        the provided cycler runs out of elements. In the cases where ``__variable__``
        is used, the data subset ``ds[var_name].sel(coords)`` gets the aesthetic
        values in `aes_dt[aes_key][var_name].sel(coords)`, however, this isn't
        always as straightforward; thus, the recommended way to get the corresponding
        aes for a specific subset is using :meth:`~arviz_plots.PlotCollection.get_aes_kwargs`

        Next let's look at the marker. We didn't provide any defaults for the marker,
        but as we specified the backend, some default values were generated for us.
        Here, we asked to encode the "field" dimension information only:

        .. jupyter-execute::

            aes_dt["marker"]

        We have a "neutral_element" variable which will be used for variables
        where the field dimension is not present and a "mapping" variable
        with a different marker value per coordinate in the field dimension,
        with all these values being different to the "neutral_element" one.
        The "y" aesthetic is very similar.

        Lastly, the "linestyle" aesthetic, which we asked to use to encode the
        chain information.

        .. jupyter-execute::

            aes_dt["linestyle"]

        As all variables have the "chain" dimension, there is no "neutral_element"
        here, and the first element in the property cycle (here the solid line "-")
        is used as part of the "chain" mapping instead of being reserved for
        variables without a "chain" dimension. Note that in such cases,
        trying to use a data variable without "chain" as dimension would
        result in an error, the mapping is not defined.

        See Also
        --------
        .PlotCollection.get_aes_kwargs
        """
        if data is None:
            data = self.data
        aes = {key: value for key, value in aes.items() if value is not False}
        extra_keys = [key for key in kwargs if key not in aes]
        if extra_keys:
            raise ValueError(
                f"Keyword arguments {extra_keys} have been passed as **kwargs but "
                "have no active aesthetic mapped to them. Keyword arguments must define "
                "values to use in their respective aesthetic mapping."
            )
        if not hasattr(self, "backend"):
            plot_bknd = import_module(".backend.none", package="arviz_plots")
        else:
            plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")
        get_default_aes = plot_bknd.get_default_aes
        ds_dict = {aes_key: xr.Dataset() for aes_key in aes}
        all_dims = set(dim for dims in aes.values() for dim in dims)
        clean_sizes = {}
        coords = {}
        for dim in all_dims:
            if dim == "__variable__":
                continue
            unique_values = np.unique(data[dim])
            clean_sizes[dim] = len(unique_values)
            if len(data[dim]) == len(unique_values):
                # preserve original order if there are no unique values
                coords[dim] = data[dim]
            else:
                coords[dim] = unique_values

        for aes_key, dims in aes.items():
            if "__variable__" in dims:
                total_aes_vals = int(
                    np.sum(
                        [
                            np.prod(
                                [
                                    clean_sizes[dim]
                                    for dim in dims
                                    if dim in set(da.dims).union(da.coords)
                                ]
                            )
                            for da in self.data.values()
                        ]
                    )
                )
                aes_vals = get_default_aes(aes_key, total_aes_vals, kwargs)
                aes_cumulative = 0
                for var_name, da in data.items():
                    aes_dims = [dim for dim in dims if dim in set(da.dims).union(da.coords)]
                    aes_raw_shape = [clean_sizes[dim] for dim in aes_dims]
                    if not aes_raw_shape:
                        ds_dict[aes_key][var_name] = np.asarray(aes_vals)[
                            aes_cumulative : aes_cumulative + 1
                        ].squeeze()
                        aes_cumulative += 1
                        continue
                    n_aes = np.prod(aes_raw_shape)
                    ds_dict[aes_key][var_name] = xr.DataArray(
                        np.array(aes_vals[aes_cumulative : aes_cumulative + n_aes]).reshape(
                            aes_raw_shape
                        ),
                        dims=aes_dims,
                        coords={dim: coords[dim] for dim in aes_dims},
                    )
                    aes_cumulative += n_aes
            else:
                aes_dims_in_var = {
                    var_name: set(dims) <= set(da.dims).union(da.coords)
                    for var_name, da in data.items()
                }
                if not any(aes_dims_in_var.values()):
                    warnings.warn(
                        f"Provided mapping for {aes_key} will only use the neutral element"
                    )
                aes_shape = [clean_sizes[dim] for dim in dims]
                total_aes_vals = int(np.prod(aes_shape))
                neutral_element_needed = not all(aes_dims_in_var.values())
                aes_vals = get_default_aes(aes_key, total_aes_vals + neutral_element_needed, kwargs)
                if neutral_element_needed:
                    neutral_element = aes_vals[0]
                    ds_dict[aes_key]["neutral_element"] = neutral_element
                    aes_vals_no_neutral = [val for val in aes_vals if val != neutral_element]
                    if aes_vals_no_neutral[0] in aes_vals_no_neutral[1:]:
                        cycle_repeat_index = aes_vals_no_neutral[1:].index(aes_vals_no_neutral[0])
                        aes_vals_no_neutral = aes_vals_no_neutral[: cycle_repeat_index + 1]
                    if aes_vals[1] == neutral_element:
                        aes_vals = [neutral_element] + aes_vals_no_neutral
                    else:
                        aes_vals = aes_vals_no_neutral
                    aes_vals = get_default_aes(
                        aes_key,
                        total_aes_vals,
                        {aes_key: aes_vals},
                    )
                ds_dict[aes_key]["mapping"] = xr.DataArray(
                    np.array(aes_vals).reshape(aes_shape),
                    dims=dims,
                    coords={dim: coords[dim] for dim in dims},
                )
        return xr.DataTree.from_dict(ds_dict)

    def get_aes_as_dataset(self, aes_key):
        """Get the values of the provided aes_key for all variables as a Dataset.

        Parameters
        ----------
        aes_key : str
            Aesthetic mapping whose values should be returned as a Dataset.
            Must be a leaf node of all groups in :attr:`~.PlotCollection.aes`

        Returns
        -------
        Dataset

        See Also
        --------
        arviz_plots.PlotCollection.update_aes_from_dataset
        """
        return self.aes[aes_key].to_dataset()

    def update_aes_from_dataset(self, aes_key, dataset):
        """Update the values of aes_key with those in the provided Dataset.

        Parameters
        ----------
        aes_key : str
            Aesthetic mapping whose values should be updated or added.
            :attr:`~.PlotCollection.aes` will contain `aes_key` as a leaf
            for all its groups, with the values provided.
        dataset : Dataset
            Dataset containing the `aes_key` values for each data variable.
            The data variables of the Dataset must match the groups of
            :attr:`~.PlotCollection.aes`

        See Also
        --------
        arviz_plots.PlotCollection.get_aes_as_dataset
        """
        self._aes_dt[aes_key] = dataset

    @property
    def facet_dims(self):
        """Dimensions over which one should loop for facetting when using this PlotCollection.

        When adding specific visuals, we might need to loop over more dimensions than these ones
        due to the defined aesthetic mappings.
        """
        return set(self.viz["plot"].dims)

    def get_viz(self, artist_name, var_name=None, sel=None, **sel_kwargs):
        """Get element from ``.viz`` that corresponds to the provided subset.

        Parameters
        ----------
        artist_name : str
        var_name : str, optional
        sel : mapping, optional
        **sel_kwargs : mapping, optional
            kwargs version of `sel`
        """
        if sel is None:
            sel = {}
        sel = sel | sel_kwargs
        out = self.viz[artist_name]
        if isinstance(out, xr.DataTree):
            out = out.dataset
            if var_name is not None:
                out = out[var_name]
        subset = sel_subset(sel, out)
        if subset:
            out = out.sel(subset)
        if isinstance(out, xr.DataArray) and out.size == 1:
            return out.item()
        return out

    def rename_visuals(self, name_dict=None, **names):
        """Rename visual data variables in the :attr:`~.PlotCollection.viz` DataTree.

        Parameters
        ----------
        name_dict, **names : mapping
            Keys are current visual names and values are desired names.
            At least one of these must be provided.
        """
        if name_dict is None:
            name_dict = names
        else:
            name_dict = names | name_dict
        self.viz = self.viz.assign(
            {
                desired_name: self.viz[current_name]
                for current_name, desired_name in name_dict.items()
            }
        ).drop_nodes(list(name_dict.keys()))

    @classmethod
    def wrap(
        cls,
        data,
        cols=None,
        col_wrap=4,
        backend=None,
        figure_kwargs=None,
        **kwargs,
    ):
        """Instantiate a PlotCollection and generate a grid iterating over subsets and wrapping.

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
        col_wrap : int or None, default 4
            Number of columns in the generated grid. If more than `col_wrap`
            plots are needed from :term:`faceting` according to `cols`,
            new rows are created. If ``None``, the number of columns is inferred
            to create a grid as close to a square as possible.
        backend : str, optional
            Plotting backend.
        figure_kwargs : mapping, optional
            Passed to :func:`~.backend.create_plotting_grid` of the chosen plotting backend.
            To add a figure title, use :meth:`~arviz_plots.PlotCollection.add_title` after
            creating the PlotCollection.
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
        if figure_kwargs is None:
            figure_kwargs = {}
        if backend is None:
            backend = rcParams["plot.backend"]
        data = concat_model_dict(data)

        n_plots, plots_per_var = process_facet_dims(data, cols)

        if col_wrap is None:
            col_wrap = int(np.ceil(np.sqrt(n_plots)))
        else:
            if not isinstance(col_wrap, (int, np.integer)):
                raise TypeError(f"col_wrap must be an int or None, got {type(col_wrap)!r}")
            if col_wrap < 1:
                raise ValueError(f"col_wrap >= 1, got {col_wrap}")

        if n_plots <= col_wrap:
            n_rows, n_cols = 1, n_plots
        else:
            div_mod = divmod(n_plots, col_wrap)
            n_rows = div_mod[0] + (div_mod[1] != 0)
            n_cols = col_wrap

        plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
        fig, ax_ary = plot_bknd.create_plotting_grid(
            n_plots, n_rows, n_cols, squeeze=False, **figure_kwargs
        )
        col_id, row_id = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        viz_dict = {}
        flat_ax_ary = ax_ary.flatten()[:n_plots]
        flat_row_id = row_id.flatten()[:n_plots]
        flat_col_id = col_id.flatten()[:n_plots]
        if "__variable__" not in cols:
            dims = cols  # use provided dim orders, not existing ones
            plots_raw_shape = []
            coords = {}
            for dim in dims:
                unique_values = np.unique(data[dim])
                plots_raw_shape.append(len(unique_values))
                if len(unique_values) == len(data[dim]):
                    # preserve original order if there are no unique values
                    coords[dim] = data[dim]
                else:
                    coords[dim] = unique_values
            viz_dict["/"] = xr.Dataset(
                {
                    "figure": np.array(fig, dtype=object),
                    "plot": (dims, flat_ax_ary.reshape(plots_raw_shape)),
                    "row_index": (dims, flat_row_id.reshape(plots_raw_shape)),
                    "col_index": (dims, flat_col_id.reshape(plots_raw_shape)),
                },
                coords=coords,
            )
        else:
            viz_dict["/"] = xr.Dataset({"figure": np.array(fig, dtype=object)})
            viz_dict["plot"] = {}
            viz_dict["row_index"] = {}
            viz_dict["col_index"] = {}
            all_dims = cols
            facet_cumulative = 0
            for var_name, da in data.items():
                coords = {}
                plots_raw_shape = []
                for dim in all_dims:
                    if dim not in set(da.dims).union(da.coords):
                        continue
                    unique_values = np.unique(da[dim])
                    plots_raw_shape.append(len(unique_values))
                    if len(unique_values) == len(data[dim]):
                        coords[dim] = data[dim]
                    else:
                        coords[dim] = unique_values
                dims = list(coords.keys())
                col_slice = (
                    slice(None, None)
                    if var_name not in plots_per_var
                    else slice(facet_cumulative, facet_cumulative + plots_per_var[var_name])
                )
                facet_cumulative += plots_per_var[var_name]
                aux_ds = xr.Dataset(
                    {
                        "plot": (
                            dims,
                            flat_ax_ary[col_slice].reshape(plots_raw_shape),
                        ),
                        "row_index": (
                            dims,
                            flat_row_id[col_slice].reshape(plots_raw_shape),
                        ),
                        "col_index": (
                            dims,
                            flat_col_id[col_slice].reshape(plots_raw_shape),
                        ),
                    },
                    coords=coords,
                )
                viz_dict["plot"][var_name] = aux_ds["plot"]
                viz_dict["row_index"][var_name] = aux_ds["row_index"]
                viz_dict["col_index"][var_name] = aux_ds["col_index"]
        viz_dt = xr.DataTree(
            viz_dict["/"],
            children={
                key: xr.DataTree(xr.Dataset(value)) for key, value in viz_dict.items() if key != "/"
            },
        )
        return cls(data, viz_dt, backend=backend, **kwargs)

    @classmethod
    def grid(
        cls,
        data,
        cols=None,
        rows=None,
        backend=None,
        figure_kwargs=None,
        **kwargs,
    ):
        """Instantiate a PlotCollection and generate a plot grid iterating over rows and columns.

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
        figure_kwargs : mapping, optional
            Passed to :func:`~.backend.create_plotting_grid` of the chosen plotting backend.
            To add a figure title, use :meth:`~arviz_plots.PlotCollection.add_title` after
            creating the PlotCollection.
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
        if figure_kwargs is None:
            figure_kwargs = {}
        if backend is None:
            backend = rcParams["plot.backend"]
        repeated_dims = [col for col in cols if col in rows]
        if repeated_dims:
            raise ValueError("The same dimension can't be used for both cols and rows.")
        data = concat_model_dict(data)

        n_cols, cols_per_var = process_facet_dims(data, cols)
        n_rows, rows_per_var = process_facet_dims(data, rows)

        n_plots = n_cols * n_rows
        plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
        fig, ax_ary = plot_bknd.create_plotting_grid(
            n_plots, n_rows, n_cols, squeeze=False, **figure_kwargs
        )
        col_id, row_id = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        viz_dict = {}
        if "__variable__" not in cols and "__variable__" not in rows:
            dims = tuple((*rows, *cols))  # use provided dim orders, not existing ones
            plots_raw_shape = []
            coords = {}
            for dim in dims:
                unique_values = np.unique(data[dim])
                plots_raw_shape.append(len(unique_values))
                if len(unique_values) == len(data[dim]):
                    coords[dim] = data[dim]
                else:
                    coords[dim] = unique_values
            viz_dict["/"] = xr.Dataset(
                {
                    "figure": np.array(fig, dtype=object),
                    "plot": (dims, ax_ary.flatten().reshape(plots_raw_shape)),
                    "row_index": (dims, row_id.flatten().reshape(plots_raw_shape)),
                    "col_index": (dims, col_id.flatten().reshape(plots_raw_shape)),
                },
                coords=coords,
            )
        else:
            viz_dict["/"] = xr.Dataset({"figure": np.array(fig, dtype=object)})
            viz_dict["plot"] = {}
            viz_dict["row_index"] = {}
            viz_dict["col_index"] = {}
            all_dims = tuple((*rows, *cols))  # use provided dim orders, not existing ones
            facet_cumulative = 0
            coords = {}
            for var_name, da in data.items():
                dims = [dim for dim in all_dims if dim in da.dims]
                plots_raw_shape = []
                dims = []
                for dim in all_dims:
                    if dim not in set(da.dims).union(da.coords):
                        continue
                    if dim in coords:
                        unique_values = coords[dim]
                    else:
                        unique_values = np.unique(da[dim])
                        if len(unique_values) == len(data[dim]):
                            coords[dim] = data[dim]
                        else:
                            coords[dim] = unique_values
                    plots_raw_shape.append(len(unique_values))
                    dims.append(dim)
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
                viz_dict["plot"][var_name] = (
                    dims,
                    ax_ary[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                )
                viz_dict["row_index"][var_name] = (
                    dims,
                    row_id[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                )
                viz_dict["col_index"][var_name] = (
                    dims,
                    col_id[row_slice, col_slice].flatten().reshape(plots_raw_shape),
                )
            viz_dict = {key: xr.Dataset(value, coords=coords) for key, value in viz_dict.items()}
        viz_dt = xr.DataTree.from_dict(viz_dict)
        return cls(data, viz_dt, backend=backend, **kwargs)

    def update_aes(self, ignore_aes=frozenset(), coords=None):
        """Update list of aesthetics after indicating ignores and extra subsets."""
        if coords is None:
            coords = {}
        aes = [aes_key for aes_key in self.aes_set if aes_key not in ignore_aes]
        aes_dims = [dim for aes_key in aes for dim in self.aes[aes_key].dims]
        all_loop_dims = self.facet_dims.union(aes_dims).difference(coords.keys())
        return aes, all_loop_dims

    def allocate_artist(
        self,
        fun_label,
        data,
        all_loop_dims,
        dim_to_idx=None,
        artist_dims=None,
        ignore_aes=frozenset(),
    ):
        """Allocate a visual in the ``viz`` DataTree."""
        if artist_dims is None:
            artist_dims = {}
        if dim_to_idx is None:
            dim_to_idx = {}
        artist_dt = xr.DataTree()
        if ignore_aes:
            artist_dt.attrs = {"ignore_aes": ignore_aes}
        for var_name, da in data.items():
            inherited_dims = [
                dim_to_idx.get(dim, dim)
                for dim in da.dims
                if (dim in all_loop_dims) or (dim in dim_to_idx)
            ]
            artist_shape = [
                da.sizes[dim_or_idx] if dim_or_idx in da.sizes else len(np.unique(da[dim_or_idx]))
                for dim_or_idx in inherited_dims
            ] + list(artist_dims.values())
            all_artist_dims = inherited_dims + list(artist_dims.keys())

            # TODO: once DataTree has a .loc attribute, this should work on .viz instead
            artist_dt[var_name] = xr.DataArray(
                np.full(artist_shape, None, dtype=object),
                dims=all_artist_dims,
                coords={
                    dim: np.unique(da[dim]) if dim in dim_to_idx.values() else da[dim]
                    for dim in inherited_dims
                },
            )
        self._viz_dt[fun_label] = artist_dt

    def get_target(self, var_name, selection):
        """Get the target that corresponds to the given variable and selection."""
        return self.get_viz("plot", var_name, selection)

    def iget_target(self, row_index, col_index):
        """Get a plot representing object from the ``.viz`` attribute by positional index.

        Parameters
        ----------
        row_index, col_index : int
            Indexes for the target plot to return.

        Notes
        -----
        At first glance, the logic of this function seems to be simplifiable to a call to the
        where method ``.where(viz["row_index"] == i & viz["col_index"] == j)`` followed by
        some checks on Dataset/DataArray inputs. However, that is not the case.
        In at least matplotlib, using `.where` with ``drop=True`` triggers a copy,
        so if on a notebook, we end up with multiple instances of basically the same plot,
        with the final tweaks after the positional indexing only applied to the last one.

        Therefore, this method uses `.where` only on the row and column indexes data which
        have integer dtype and extracts the variable and indexes from there. This allows
        using sel/isel selection on to retrieve the plot objects.
        """
        if "plot" in self.viz.data_vars:
            row_da = self.viz["row_index"]
            col_da = self.viz["col_index"]
            if row_index < 0:
                row_index = int(row_da.max() + 1 + row_index)
            if col_index < 0:
                col_index = int(col_da.max() + 1 + col_index)
            condition = (row_da == row_index) & (col_da == col_index)
            if not condition.any():
                raise ValueError(
                    f"Mo match found for provided indexes (row: {row_index}, column: {col_index}). "
                    "Check indexes are within the grid size and don't represent an empty plot"
                )
            return self.viz["plot"].isel(condition.argmax(...)).item()
        row_ds = self.viz["row_index"].dataset
        col_ds = self.viz["col_index"].dataset
        if row_index < 0:
            row_index = int(row_ds.max().to_array().max() + 1 + row_index)
        if col_index < 0:
            col_index = int(col_ds.max().to_array().max() + 1 + col_index)
        condition = (row_ds == row_index) & (col_ds == col_index)
        var_condition = condition.any().to_array()
        if not var_condition.any():
            raise ValueError(
                f"Mo match found for provided indexes (row: {row_index}, column: {col_index}). "
                "Check indexes are within the grid size and don't represent an empty plot"
            )
        target_var = var_condition.coords["variable"][var_condition.argmax("variable")].item()
        return self.viz["plot"][target_var].isel(condition[target_var].argmax(...)).item()

    def get_aes_kwargs(self, aes, var_name, selection):
        """Get the aesthetic mappings for the given variable and selection as a dictionary.

        Parameters
        ----------
        aes : list
            List of aesthetic keywords whose values should be retrieved. Values are taken
            from the ``aes`` attribute: groups as the elements in `aes` argument,
            variable `var_name` argument if present, otherwise "mapping" or "neutral_element"
            and `selection` coordinate/dimension subset.

            :class:`.PlotCollection` considers aesthetics starting with "overlay"
            a special aesthetic keyword to indicate visual elements with potentially
            identical properties should be overlaid.
            Thus, if "overlay" or "overlay_xyz" are an element of the `aes` argument,
            it is skipped, no value is attempted to be retrieved and it isn't present
            as key in the returned output either.
        var_name : str
        selection : dict

        Returns
        -------
        dict
            Mapping of aesthetic keywords to the values corresponding to the provided
            `var_name` and `selection`.

        See Also
        --------
        .PlotCollection.generate_aes_dt
        """
        aes_kwargs = {}
        for aes_key in aes:
            if aes_key.startswith("overlay"):
                continue
            aes_ds = self.aes[aes_key]
            if var_name in aes_ds.data_vars:
                aes_kwargs[aes_key] = subset_ds(aes_ds, var_name, selection)
            else:
                if all(dim in selection for dim in aes_ds["mapping"].dims):
                    aes_kwargs[aes_key] = subset_ds(aes_ds, "mapping", selection)
                elif "neutral_element" in aes_ds.data_vars:
                    aes_kwargs[aes_key] = subset_ds(aes_ds, "neutral_element", {})
                else:
                    raise ValueError(
                        f"{aes_key} has no neutral element initialized but "
                        f"{var_name} needs a neutral element."
                    )
        return aes_kwargs

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
        data : Dataset or DataArray, optional
            Data to be subsetted at each iteration and to pass to `fun` as first positional
            argument. If `data` is a DataArray it must be named.
            Defaults to the data used to initialize the ``PlotCollection``.
        coords : mapping, optional
            Dictionary of {coordinate names : coordinate values} that should
            be used to subset the aes, data and viz objects before any faceting
            or aesthetics mapping is applied.
        ignore_aes : set or "all", optional
            Set of aesthetics present in ``aes`` that should be ignore for this
            ``map`` call. The string "all" is also valid to indicate all aesthetics
            should be ignored, thus taking only facetting into account.
        subset_info : boolean, default False
            Add the subset info from :func:`arviz_base.xarray_sel_iter` to
            the keyword arguments passed to `fun`. If true, then `fun` must
            accept the keyword arguments ``var_name``, ``sel`` and ``isel``.
            Moreover, if those were to be keys present in `**kwargs` their
            values in `**kwargs` would be ignored.
        store_artist : boolean, default True
        artist_dims : mapping of {hashable : int}, optional
            Dictionary of sizes for proper allocation and storage when using
            ``map`` with functions that return an array of :term:`visual`.
        **kwargs : mapping, optional
            Keyword arguments passed as is to `fun`. Values within `**kwargs`
            with :class:`~xarray.DataArray` of :class:`~xarray.Dataset` type
            will be subsetted on the current selection (if possible) before calling `fun`.
            Slicing with dims and coords is applied to the relevant subset present in the
            xarray object so dimensions with mapped asethetics not being present is not an issue.
            However, using Datasets that don't contain all the variable names in `data`
            will raise an error.
        """
        if coords is None:
            coords = {}
        if fun_label is None:
            fun_label = fun.__name__
        if isinstance(ignore_aes, str) and ignore_aes == "all":
            ignore_aes = self.aes_set

        data = self.data if data is None else data
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        if not isinstance(data, xr.Dataset):
            raise TypeError("data argument must be an xarray.Dataset")

        aes, all_loop_dims = self.update_aes(ignore_aes, coords)
        dim_to_idx = {
            data[idx].dims[0]: idx
            for idx in data.coords
            if (idx in all_loop_dims) and (idx not in data.dims)
        }
        for idx in dim_to_idx.values():
            if idx not in data.xindexes:
                data = data.set_xindex(idx)
        skip_dims = {
            dim for dim in data.dims if (dim not in all_loop_dims) and (dim not in dim_to_idx)
        }
        plotters = xarray_sel_iter(data, skip_dims=skip_dims, dim_to_idx=dim_to_idx)
        if store_artist:
            self.allocate_artist(
                fun_label=fun_label,
                data=data,
                all_loop_dims=all_loop_dims,
                dim_to_idx=dim_to_idx,
                artist_dims=artist_dims,
                ignore_aes=ignore_aes,
            )

        for var_name, sel, isel in plotters:
            da = data[var_name].sel(sel)
            try:
                if np.all(np.isnan(da)):
                    continue
            except TypeError:
                pass
            sel_plus = {**sel, **coords}
            target = self.get_target(var_name, sel_plus)

            aes_kwargs = self.get_aes_kwargs(aes, var_name, sel_plus)

            fun_kwargs = {
                **aes_kwargs,
                **{
                    key: process_kwargs_subset(values, var_name, sel)
                    for key, values in kwargs.items()
                },
            }
            if subset_info:
                fun_kwargs = {**fun_kwargs, "var_name": var_name, "sel": sel, "isel": isel}

            aux_artist = fun(da, target=target, **fun_kwargs)
            if store_artist:
                if np.size(aux_artist) == 1:
                    aux_artist = np.squeeze(aux_artist)
                self.store_in_artist_da(aux_artist, fun_label, var_name, sel)

    def store_in_artist_da(self, aux_artist, fun_label, var_name, sel):
        """Store the visual object of `var_name`+`sel` combination in `fun_label` variable."""
        self.viz[fun_label][var_name].loc[sel] = aux_artist

    def add_title(self, text, *, color=None, size=None, **artist_kws):
        """Add a title to the figure.

        Parameters
        ----------
        text : str
            The title text.
        color : str or tuple, optional
            Color of the title text.
        size : float, optional
            Font size of the title.
        **artist_kws : mapping, optional
            Additional keyword arguments passed to the backend title function.

        Returns
        -------
        title : object
            The title object for the backend.

        Examples
        --------
        Add a title after creating a plot:

        .. jupyter-execute::

            import arviz_base as azb
            import arviz_plots as azp

            data = azb.load_arviz_data("centered_eight")
            pc = azp.plot_dist(data)
            pc.add_title("Posterior Distributions")

        Add a colored title with custom size:

        .. jupyter-execute::

            pc = azp.plot_trace(data, var_names=["mu"])
            pc.add_title("MCMC Trace", color="darkblue", size=16)
        """
        if "figure" not in self.viz:
            raise ValueError("No figure found to add title to")

        plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")
        fig = self.viz["figure"].item()
        
        new_fig, title_obj = plot_bknd.set_figure_title(
            fig, text, color=color, size=size, **artist_kws
        )
        
        # bokeh returns a new column layout, so we need to update the stored figure
        if new_fig is not fig:
            self.viz["figure"] = xr.DataArray(new_fig)
        
        self.viz["figure_title"] = xr.DataArray(title_obj)
        
        return title_obj

    def add_legend(
        self,
        dim,
        aes=None,
        visual_kwargs=None,
        title=None,
        text_only=False,
        # position=(0, -1),  # TODO: add argument
        labeller=None,
        **kwargs,
    ):
        """Add a legend for the given visual/aesthetic to the plot.

        Warnings
        --------
        This method is still in early stages of experimentation and anything beyond
        the basic usage ``add_legend("dim_name")`` will probably change in breaking ways.

        Parameters
        ----------
        dim : hashable or iterable of hashable
            Dimension or dimensions for which to generate the legend.
            The pseudo-dimension ``__variable__`` is allowed too.
            It should have at least one :term:`aesthetic mapped <aesthetic mapping>` to it.
            Only the mappings that match will be taken into account; if a legend is requested
            for the "chain" dimension but there is only one aesthetic mapping
            for ("chain", "group") no legend can be generated.
        aes : str or iterable of str, optional
            Specific aesthetics to take into account when generating the legend.
            They should all be mapped to `dim`. Defaults to all aesthetics matching
            that mapping with the exception "x" and "y" which are never included.
        visual_kwargs : mapping, optional
            Keyword arguments passed to the backend visual function used to
            generate the miniatures in the legend.
        title : str, optional
            Legend title. Defaults to `dim`.
        text_only : bool, optional
            If True, creates a text-only legend without graphical markers.
        labeller : labeller instance, optional
            Labeller to generate the legend entries
        position : (int, int), default (0, -1)
        **kwargs : mapping, optional
            Keyword arguments passed to the backend function that generates the legend.

        Returns
        -------
        legend : object
            The corresponding legend object for the backend of the ``PlotCollection``.
        """
        if isinstance(dim, str):
            dim = (dim,)
        else:
            dim = tuple(dim)
        update_visuals = False
        if "legendgroup" not in self.aes.children and self.backend == "plotly":
            # TODO: keep if to avoid duplicating legendgroup but don't make it plotly specific
            # also, should we add an "interactive" argument to disable this behaviour
            # even if it would be possible?
            update_visuals = True
            self.update_aes_from_dataset(
                "legendgroup", self.generate_aes_dt({"legendgroup": dim})["legendgroup"].dataset
            )
        dim_str = ", ".join(("variable" if d == "__variable__" else d for d in dim))
        if title is None:
            title = dim_str
        aes_mappings = {
            aes_key: list(ds.dims) + ([] if "mapping" in ds.data_vars else ["__variable__"])
            for aes_key, ds in self.aes.children.items()
        }
        valid_aes = [
            aes_key for aes_key, aes_dims in aes_mappings.items() if set(dim) == set(aes_dims)
        ]
        if not valid_aes:
            raise ValueError(
                f"Legend can't be generated. Found no aesthetics mapped to dimension {dim}. "
                f"Existing mappings are {aes_mappings}."
            )
        if aes is None:
            aes = [aes_key for aes_key in valid_aes if aes_key not in ("x", "y")]
        elif isinstance(aes, str):
            aes = [aes]

        if labeller is None:
            labeller = BaseLabeller()

        sample_aes_ds = self.aes[aes[0]].dataset
        subset_iterator = list(xarray_sel_iter(sample_aes_ds, skip_dims=set()))
        if "__variable__" in dim:
            label_list = [labeller.make_label_flat(*subset) for subset in subset_iterator]
        else:
            label_list = [
                labeller.sel_to_str(sel, isel) if var_name == "mapping" else "∅"
                for var_name, sel, isel in subset_iterator
            ]
        if text_only:
            kwarg_list = [{} for _ in subset_iterator]
            visual_kwargs = {"linestyle": "none", "linewidth": 0, "color": "none"}
        else:
            kwarg_list = [
                self.get_aes_kwargs(aes, var_name, sel) for var_name, sel, _ in subset_iterator
            ]

        plot_bknd = import_module(f".backend.{self.backend}", package="arviz_plots")

        legend_title = None if text_only else title

        self.viz[f"legend/{dim_str}"] = plot_bknd.legend(
            self,
            kwarg_list,
            label_list,
            title=legend_title,
            visual_kwargs=visual_kwargs,
            legend_dim=dim,
            update_visuals=update_visuals,
            **kwargs,
        )
