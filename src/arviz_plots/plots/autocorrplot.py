"""Autocorrelation plot code."""

from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset, rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import process_group_variables_coords
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
    """Generate autocorrelation plots for the given dataset."""
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}

    # Default max lag to 100
    if max_lag is None:
        max_lag = 100

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    print("Processed distribution:", distribution)

    # Compute autocorrelation for each variable and chain
    acf_data = []
    for var in distribution.data_vars:
        var_data = distribution[var]
        print(f"Processing variable: {var}")
        if "chain" in var_data.dims and not combined:
            for chain in var_data.chain.values:
                chain_data = var_data.sel(chain=chain)
                print(f"Processing chain: {chain}")
                # Ensure sample_dims are valid for the current data
                valid_sample_dims = [dim for dim in sample_dims if dim in chain_data.dims]
                if not valid_sample_dims:
                    raise ValueError(
                        f"None of the sample_dims {sample_dims} present in data for {var}"
                    )
                # Compute autocorrelation
                acf = chain_data.azstats.autocorr(dims=valid_sample_dims)
                print(f"Autocorrelation result for {var}, chain {chain}: {acf}")
                # Add chain and variable as coordinates
                acf = acf.assign_coords({"chain": chain, "variable": var})
                acf_data.append(acf)
        else:
            # Ensure sample_dims are valid for the current data
            valid_sample_dims = [dim for dim in sample_dims if dim in var_data.dims]
            if not valid_sample_dims:
                raise ValueError(f"None of the sample_dims {sample_dims} present in data for {var}")
            # Compute autocorrelation
            acf = var_data.azstats.autocorr(dims=valid_sample_dims)
            print(f"Autocorrelation result for {var}: {acf}")
            # Add variable as a coordinate
            acf = acf.assign_coords({"variable": var})
            acf_data.append(acf)

    # Combine all autocorrelation results into a single DataArray
    acf_data = xr.concat(acf_data, dim="variable")
    print("Combined acf_data:", acf_data)
    print("Shape of acf_data:", acf_data.shape)

    # Convert acf_data to Dataset with the correct variable name
    acf_dataset = acf_data.to_dataset(name="autocorr")
    print("acf_dataset:", acf_dataset)
    print("Variables in acf_dataset:", list(acf_dataset.data_vars))  # Should include 'autocorr'

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        # Set up faceting
        pc_kwargs.setdefault("col_wrap", 4)
        pc_kwargs.setdefault("cols", ["__variable__"])
        pc_kwargs.setdefault("rows", ["chain"] if "chain" in acf_dataset.dims else [])

        # Calculate the number of plots
        n_plots = len(acf_dataset.variable) * (
            len(acf_dataset.chain) if "chain" in acf_dataset.dims else 1
        )
        print("Number of plots:", n_plots)

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
        pc_kwargs["aes"].setdefault("x", ["lag"])  # lag is on the x-axis
        pc_kwargs["aes"].setdefault("y", ["autocorr"])  # autocorr is on the y-axis

        if not combined and "chain" in acf_dataset.dims:
            pc_kwargs["aes"].setdefault("color", ["chain"])

        plot_collection = PlotCollection.grid(
            acf_dataset,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    aes_map.setdefault("line", plot_collection.aes_set)

    if labeller is None:
        labeller = BaseLabeller()

    # Plot autocorrelation lines
    line_kwargs = plot_kwargs.get("line", {}).copy()
    line_kwargs.setdefault("linewidth", 1.5)
    line_kwargs.setdefault("color", "#1f77b4")

    plot_collection.map(
        line_x,
        "autocorr",
        data=acf_dataset,
        ignore_aes=plot_collection.aes_set - {"x", "y", "color"},
        **line_kwargs,
    )

    # Add reference line at 0
    ref_line_kwargs = plot_kwargs.get("reference_line", {}).copy()
    ref_line_kwargs.setdefault("color", "gray")
    ref_line_kwargs.setdefault("linewidth", 1)
    ref_line_kwargs.setdefault("linestyle", "--")

    zero_line = xr.DataArray(np.zeros(max_lag), dims=["lag"], coords={"lag": np.arange(max_lag)})
    zero_line = zero_line.to_dataset(name="autocorr")

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
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=plot_collection.aes_set,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    if plot_kwargs.get("remove_axis", True) is not False:
        plot_collection.map(
            remove_axis, store_artist=False, axis="y", ignore_aes=plot_collection.aes_set
        )

    return plot_collection
