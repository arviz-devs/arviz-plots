"""Utilities for batteries included plots."""
import warnings
from copy import copy
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.utils import _var_names

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
    if isinstance(data, xr.Dataset):
        return data
    if hasattr(data, "name") and data.name == group:
        return data.ds
    try:
        data = data[group]
    except KeyError:
        if not allow_missing:
            raise
        return None
    if isinstance(data, xr.Dataset):
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


def _references_to_dataset(references, ds, sample_dims):
    """Generate an :class:`~xarray.Dataset` compatible with `ds` from `references`.

    Parameters
    ----------
    references : scalar, array-like, dict, DataArray
        References to cast into a compatible dataset.
    ds : Dataset
    sample_dims : iterable of hashable, optional

    Returns
    -------
    Dataset
        A Dataset with a subset of the variables, dimensions and coordinate names in `ds`,
        with only an extra dimension "ref_line_dim" added when multiple references are requested
        for one or some of the variables.
    """
    if isinstance(references, xr.Dataset):
        return references
    if isinstance(references, xr.DataArray):
        name = references.name
        if name is not None:
            if name not in ds.data_vars:
                warnings.warn(f"{name} not available in {ds.data_vars}", UserWarning)
            return references.to_dataset()
        references = references.values
    if np.isscalar(references):
        aux_ds = (
            ds if not sample_dims else ds.isel({dim: 0 for dim in sample_dims if dim in ds.dims})
        )
        return xr.full_like(aux_ds, references, dtype=np.array(references).dtype)
    if isinstance(references, list | tuple | np.ndarray):
        references = {var_name: references for var_name in ds.data_vars}
    if isinstance(references, dict):
        ref_dict = {}
        for var_name, da in ds.items():
            if var_name not in references:
                continue
            ref_values = references[var_name]
            sizes = {dim: length for dim, length in da.sizes.items() if dim not in sample_dims}
            ref_dict[var_name] = xr.DataArray(
                np.full(list(sizes.values()) + [np.size(ref_values)], ref_values),
                dims=list(sizes) + ["ref_line_dim"],
                coords={"ref_line_dim": np.arange(np.size(ref_values))}
                | {
                    coord_name: coord_da
                    for coord_name, coord_da in da.coords.items()
                    if coord_da.dims[0] not in sample_dims
                },
            )
        return xr.Dataset(ref_dict)
    raise TypeError("Unrecognized input type for `references`")


def add_reference_lines(
    plot_collection,
    references,
    orientation="vertical",
    aes_map=None,
    plot_kwargs=None,
    backend=None,
    sample_dims=None,
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
    sample_dims : list, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``

    Returns
    -------
    plot_collection : PlotCollection
        Plot collection with the reference lines added.

    Examples
    --------
    Add reference line at value 0.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_dist, add_reference_lines, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> pc = plot_dist(
        >>>     dt,
        >>>     kind="ecdf",
        >>>     var_names=["mu"],
        >>> )
        >>> add_reference_lines(pc, references=0)
    """
    if backend is None:
        backend = plot_collection.backend
    if plot_kwargs is None:
        plot_kwargs = {}
    if aes_map is None:
        aes_map = {}
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    plot_func = vlines if orientation == "vertical" else hlines

    _, ref_aes, ref_ignore = filter_aes(plot_collection, aes_map, "reference", sample_dims)
    ref_kwargs = copy(plot_kwargs.get("reference", {}))

    if "color" not in ref_aes:
        ref_kwargs.setdefault("color", "gray")
    if "linestyle" not in ref_aes:
        ref_kwargs.setdefault("linestyle", plot_bknd.get_default_aes("linestyle", 2, {})[1])
    ref_dt = _references_to_dataset(references, plot_collection.data, sample_dims=sample_dims)
    artist_dims = (
        {"ref_line_dim": ref_dt.sizes["ref_line_dim"]} if "ref_line_dim" in ref_dt.sizes else None
    )
    plot_collection.map(
        plot_func,
        "ref_lines",
        data=ref_dt,
        ignore_aes=ref_ignore,
        artist_dims=artist_dims,
        **ref_kwargs,
    )
    return plot_collection
