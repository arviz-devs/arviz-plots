"""Autocorrelation plot code."""

from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset, rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.base.core import _CoreBase

from arviz_plots.plot_collection import PlotCollection, process_facet_dims
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import labelled_title, line_x, remove_axis


def plot_autocorr(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    max_lag=None,
    combined=False,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
):
    """Generate autocorrelation plots for the given dataset.

    Parameters
    ----------
    dt : xarray.Dataset
        The dataset containing the variables to plot.
    var_names : list of str, optional
        Names of variables to include in the plot.
    filter_vars : str, optional
        Filter to apply to variable names.
    group : str, default="posterior"
        The group in the dataset to use for plotting.
    coords : dict, optional
        Coordinates to subset the dataset.
    max_lag : int, optional
        Maximum lag to compute autocorrelation for.
    combined : bool, default=False
        Whether to combine chains when computing autocorrelation.
    sample_dims : list of str, optional
        Dimensions to treat as sample dimensions.
    plot_collection : PlotCollection, optional
        Existing plot collection to use.
    backend : str, optional
        Backend to use for plotting.
    labeller : BaseLabeller, optional
        Labeller to use for plot labels.
    aes_map : dict, optional
        Mapping of aesthetics to variables.
    plot_kwargs : dict, optional
        Additional keyword arguments for the plot.
    pc_kwargs : dict, optional
        Additional keyword arguments for the plot collection.

    Returns
    -------
    plot_collection : PlotCollection
        The plot collection containing the autocorrelation plots.
    """
    dt = convert_to_dataset(dt, group="posterior")

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}

    # Default max lag to 100 or max length of chain
    if max_lag is None:
        max_lag = 100

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    # Convert xarray.Dataset to numpy array
    distribution_array = distribution.to_array().values

    # Calculate lags up to max_lag
    lags = np.arange(max_lag)

    # Calculate autocorrelation from arviz_stats autocorr computation
    core_base = _CoreBase()
    acf_data = core_base.autocorr(distribution_array, axis=-1)[..., :max_lag]

    dims = list(distribution.dims) + ["lag"]

    # Ensure correct dimension sizes
    coords = {}
    for dim in distribution.dims:
        expected_size = acf_data.shape[dims.index(dim)]
        if dim in distribution.coords and len(distribution.coords[dim]) == expected_size:
            coords[dim] = distribution.coords[dim]
        else:
            coords[dim] = np.arange(expected_size)

    # Add lag coordinate
    coords["lag"] = np.arange(acf_data.shape[-1])

    # Adjust the 'lag' coordinate to match the size of the 'lag' dimension in acf_data
    lag_size = acf_data.shape[-1]  # Size of the 'lag' dimension in acf_data
    coords["lag"] = np.arange(lag_size)  # Update 'lag' coordinate to match the size

    # Handle the case where the dimension sizes have changed
    for dim in distribution.dims:
        if dim in coords and acf_data.shape[dims.index(dim)] != len(coords[dim]):
            # Adjust the dimension coordinate to match the size of the dimension in acf_data
            coords[dim] = np.arange(acf_data.shape[dims.index(dim)])

    acf_data = xr.DataArray(
        acf_data,
        dims=dims,
        coords=coords,
    )

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        # Convert DataArray to Dataset for compatibility with process_facet_dims
        pc_data = acf_data.to_dataset(name="autocorr")
        if "column" not in pc_data.dims:
            pc_data = pc_data.expand_dims(column=["autocorr"])
        print(pc_data)

        # Set default columns and rows for faceting
        pc_kwargs.setdefault("cols", ["__variable__"])
        pc_kwargs.setdefault("rows", list(set(pc_data.dims) - {"__variable__", "lag", "chain"}))

        # Calculate the number of plots
        n_plots, plots_per_var = process_facet_dims(pc_data, pc_kwargs["cols"])

        # Set up figure size
        figsize = pc_kwargs.get("plot_grid_kws", {}).get("figsize", None)
        if figsize is None:
            col_wrap = pc_kwargs.get("col_wrap", 4)
            if n_plots <= col_wrap:
                n_rows, n_cols = 1, n_plots
            else:
                div_mod = divmod(n_plots, col_wrap)
                n_rows = div_mod[0] + (div_mod[1] != 0)
                n_cols = col_wrap
            figsize = plot_bknd.scale_fig_size(
                figsize,
                rows=n_rows,
                cols=n_cols,
            )

        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["plot_grid_kws"]["figsize"] = figsize
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("y", ["lag"])

        if not combined and "chain" in distribution.dims:
            pc_kwargs["aes"].setdefault("color", ["chain"])

        plot_collection = PlotCollection.grid(
            pc_data,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    aes_map.setdefault("line", plot_collection.aes_set)

    if labeller is None:
        labeller = BaseLabeller()

    # Convert acf_data to Dataset for compatibility with plot_collection.map
    acf_dataset = acf_data.to_dataset(name="autocorr")
    print("acf data is : ", acf_data)

    # Plot autocorrelation lines
    line_kwargs = plot_kwargs.get("line", {}).copy()
    line_kwargs.setdefault("linewidth", 1.5)  # Use 'linewidth' instead of 'width'
    line_kwargs.setdefault("color", "#1f77b4")
    print("line_kwargs:", line_kwargs)
    print("aes_map:", aes_map)

    acf_dims, acf_aes, acf_ignore = filter_aes(plot_collection, aes_map, "line", sample_dims)

    print(acf_dataset)
    print(acf_ignore)

    plot_collection.map(
        line_x,
        "autocorr",
        data=acf_dataset,
        ignore_aes=acf_ignore,
        **line_kwargs,
    )

    # Add reference line at 0
    ref_line_kwargs = plot_kwargs.get("reference_line", {}).copy()
    ref_line_kwargs.setdefault("color", "gray")
    ref_line_kwargs.setdefault("linewidth", 1)  # Use 'linewidth' instead of 'width'
    ref_line_kwargs.setdefault("linestyle", "--")

    # Create zero line as a proper DataArray
    zero_line = xr.DataArray(np.zeros(max_lag), dims=["lag"], coords={"lag": np.arange(max_lag)})

    # Convert to dataset with the same variable name as the main data
    zero_line = zero_line.to_dataset(name="autocorr")

    if "column" not in zero_line.dims:
        zero_line = zero_line.expand_dims(column=["autocorr"])

    plot_collection.map(
        line_x,
        "reference_line",
        data=zero_line,
        ignore_aes=plot_collection.aes_set,
        **ref_line_kwargs,
    )

    # Add titles for each plot
    title_kwargs = plot_kwargs.get("title", {}).copy()
    if title_kwargs is not False:
        _, title_aes, title_ignore = filter_aes(plot_collection, aes_map, "title", sample_dims)
        if "color" not in title_aes:
            title_kwargs.setdefault("color", "black")
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    if plot_kwargs.get("remove_axis", True) is not False:
        plot_collection.map(
            remove_axis, store_artist=False, axis="y", ignore_aes=plot_collection.aes_set
        )

    return plot_collection
