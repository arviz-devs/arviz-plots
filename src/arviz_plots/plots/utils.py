"""Utilities for batteries included plots."""

from copy import copy
from importlib import import_module

import numpy as np
from arviz_base.utils import _var_names
from xarray import DataArray, Dataset

from arviz_plots.plot_collection import concat_model_dict, process_facet_dims
from arviz_plots.visuals import hlines, vlines


def get_group(data, group, allow_missing=False):
    """Get a group from a Datatree or Dataset if possible and return a Dataset.

    Also supports InferenceData or dictionaries of Datasets.

    Parameters
    ----------
    data : DataTree, Dataset, InferenceData or mapping of {str : Dataset}
        Object from which to extract `group`
    group : hashable
        Id to be extracted. It is checked against the ``name`` attribute
        and attempted to use as key to get the `group` item from `data`
    allow_missing : bool, default False
        Return ``None`` if `group` can't be extracted instead of raising an error.

    Returns
    -------
    Dataset

    Raises
    ------
    KeyError
        If unable to access `group` from `data` and ``allow_missing=False``.
    """
    if isinstance(data, Dataset):
        return data
    if hasattr(data, "name") and data.name == group:
        return data.ds
    try:
        data = data[group]
    except KeyError:
        if not allow_missing:
            raise
        return None
    if isinstance(data, Dataset):
        return data
    return data.ds


def process_group_variables_coords(dt, group, var_names, filter_vars, coords, allow_dict=True):
    """Process main input arguments of batteries included plotting functions."""
    if coords is None:
        coords = {}
    if isinstance(dt, dict) and not allow_dict:
        raise ValueError("Input data as dictionary not supported")
    if isinstance(dt, dict):
        distribution = {}
        for key, value in dt.items():
            var_names = _var_names(var_names, get_group(value, group), filter_vars)
            distribution[key] = (
                get_group(value, group).sel(coords)
                if var_names is None
                else get_group(value, group)[var_names].sel(coords)
            )
        distribution = concat_model_dict(distribution)
    else:
        distribution = get_group(dt, group)
        var_names = _var_names(var_names, distribution, filter_vars)
        if var_names is not None:
            distribution = distribution[var_names]
        distribution = distribution.sel(coords)
    return distribution


def filter_aes(pc, aes_map, artist, sample_dims):
    """Split aesthetics and get relevant dimensions.

    Returns
    -------
    artist_dims : list
        Dimensions that should be reduced for this artist.
        That is, all dimensions in `sample_dims` that are not
        mapped to any aesthetic.
    artist_aes : iterable
    ignore_aes : set
    """
    artist_aes = aes_map.get(artist, {})
    pc_aes = pc.aes_set
    ignore_aes = set(pc_aes).difference(artist_aes)
    _, all_loop_dims = pc.update_aes(ignore_aes=ignore_aes)
    artist_dims = [dim for dim in sample_dims if dim not in all_loop_dims]
    return artist_dims, artist_aes, ignore_aes


def set_wrap_layout(pc_kwargs, plot_bknd, ds):
    """Set the figure size and handle column wrapping.

    Parameters
    ----------
    pc_kwargs : dict
        Plot collection kwargs
    plot_bknd : str
        Backend for plotting
    ds : Dataset
        Dataset to be plotted
    """
    figsize = pc_kwargs["plot_grid_kws"].get("figsize", None)
    figsize_units = pc_kwargs["plot_grid_kws"].get("figsize_units", "inches")
    pc_kwargs.setdefault("col_wrap", 4)
    col_wrap = pc_kwargs["col_wrap"]
    if figsize is None:
        num_plots = process_facet_dims(ds, pc_kwargs["cols"])[0]
        if num_plots < col_wrap:
            cols = num_plots
            rows = 1
        else:
            div_mod = divmod(num_plots, col_wrap)
            rows = div_mod[0] + (div_mod[1] != 0)
            cols = col_wrap
        figsize = plot_bknd.scale_fig_size(
            figsize,
            rows=rows,
            cols=cols,
            figsize_units=figsize_units,
        )
        figsize_units = "dots"

    pc_kwargs["plot_grid_kws"]["figsize"] = figsize
    pc_kwargs["plot_grid_kws"]["figsize_units"] = figsize_units
    return pc_kwargs


def set_grid_layout(pc_kwargs, plot_bknd, ds, num_rows=None, num_cols=None):
    """Set the figure size for the given number of rows and columns.

    Parameters
    ----------
    pc_kwargs : dict
        Plot collection kwargs
    plot_bknd : str
        Backend for plotting
    ds : Dataset
        Dataset to be plotted
    num_rows, num_cols : int, optional
        Take the number of rows or columns as the provided one irrespective
        of pc_kwargs
    """
    figsize = pc_kwargs["plot_grid_kws"].get("figsize", None)
    figsize_units = pc_kwargs["plot_grid_kws"].get("figsize_units", "inches")
    if figsize is None:
        if num_cols is None:
            num_cols = process_facet_dims(ds, pc_kwargs["cols"])[0]
        if num_rows is None:
            num_rows = process_facet_dims(ds, pc_kwargs["rows"])[0]
        figsize = plot_bknd.scale_fig_size(
            figsize,
            rows=num_rows,
            cols=num_cols,
            figsize_units=figsize_units,
        )
        figsize_units = "dots"

    pc_kwargs["plot_grid_kws"]["figsize"] = figsize
    pc_kwargs["plot_grid_kws"]["figsize_units"] = figsize_units
    return pc_kwargs


def add_reference_lines(
    plot_collection,
    references,
    orientation="vertical",
    aes_map=None,
    plot_kwargs=None,
    backend=None,
    data_vars=None,
):
    """Add reference lines.

    This function adds lines to a plot collection based on the provided
    references. It supports both vertical and horizontal lines, depending on the
    specified orientation.

    Parameters
    ----------
    plot_collection : PlotCollection
        Plot collection to which the reference lines will be added.
    references : int, float, tuple, list or dict
        Reference values to be plotted as lines.
    orientation : str, default "vertical"
        The orientation of the reference lines, either "vertical" or "horizontal".
    aes_map : dict, optional
        A dictionary mapping aesthetics to their corresponding variables.
    plot_kwargs : dict, optional
        A dictionary containing the plot arguments.
    backend : str, optional
        The backend used for plotting.
    data_vars : list, optional
        A list of data variables to be used for plotting.
    """
    if backend is None:
        backend = plot_collection.backend
    if plot_kwargs is None:
        plot_kwargs = {}
    if aes_map is None:
        aes_map = {}
    if data_vars is None:
        data_vars = plot_collection.data.data_vars

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    plot_func = vlines if orientation == "vertical" else hlines

    _, ref_aes, ref_ignore = filter_aes(plot_collection, aes_map, "reference", "sample")
    ref_kwargs = copy(plot_kwargs.get("reference", {}))

    if "color" not in ref_aes:
        ref_kwargs.setdefault("color", "gray")
    if "linestyle" not in ref_aes:
        ref_kwargs.setdefault("linestyle", plot_bknd.get_default_aes("linestyle", 2, {})[1])

    if isinstance(references, dict):
        for key, value in references.items():
            if key in data_vars:
                ref_dt = Dataset({key: DataArray(np.asarray(value))})
                plot_collection.map(
                    plot_func, "ref_lines", data=ref_dt, ignore_aes=ref_ignore, **ref_kwargs
                )
    else:
        ref_dt = Dataset({var: DataArray(np.asarray(references)) for var in data_vars})
        plot_collection.map(
            plot_func, "ref_lines", data=ref_dt, ignore_aes=ref_ignore, **ref_kwargs
        )
