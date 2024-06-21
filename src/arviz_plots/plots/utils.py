"""Utilities for batteries included plots."""

import xarray as xr
from arviz_base.utils import _var_names
from xarray import Dataset

from arviz_plots.plot_collection import concat_model_dict


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
        if allow_missing:
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


# WIP: dropping bin_midpoints calculation and keeping bin_edges
def restructure_hist_data(hist_dict):
    """Restructure histogram DataArrays from xarray-einstats.numba.histogram() into a Dataset.

    Dataset returned resembles the KDE Dataset returned by azstats.kde(). The 'bin' dimension is
    renamed to 'hist_dim', coordinates 'left_edge' and 'right_edge' are dropped. Bin left and right
    edges are broadcasted to histogram DataArray shape, and concatenated along a new dimension
    'plot_axis' with bin_heights along coord 'y' and left and right edges along coords 'l_e' and
    'r_e'.

    Parameters
    ----------
    hist_dict : dict
        A dictionary containing histogram DataArrays, where keys are variable names and values are
        histogram DataArrays.

    Returns
    -------
    restructured_hist_ds : Dataset
        A Dataset containing restructured histogram DataArrays, where each variable corresponds to
        a histogram DataArray

    """
    restructured_hist_dict = {}
    for var_name, hist in hist_dict.items():
        # restructuring hist to a DataArray by renaming bin dimension
        bin_heights_darr = hist.rename({"bin": "hist_dim"}).drop_vars(["left_edges", "right_edges"])

        left_edges_darr = hist.rename({"bin": "hist_dim"}).coords["left_edges"]
        left_edges_darr = left_edges_darr.drop_vars(["left_edges", "right_edges"])
        right_edges_darr = hist.rename({"bin": "hist_dim"}).coords["right_edges"]
        right_edges_darr = right_edges_darr.drop_vars(["left_edges", "right_edges"])

        # broadcasting left_edges_darr and right_edges_darr to fit bin_heights_darr shape
        bin_heights_darr, left_edges_darr, right_edges_darr = xr.broadcast(
            bin_heights_darr, left_edges_darr, right_edges_darr
        )

        # concatenating bin_heights, left_edges, and right_edges along the new plot-axis dimension
        hist_darr = xr.concat(
            [left_edges_darr, right_edges_darr, bin_heights_darr], dim="plot_axis"
        )

        # assigning plot_axis coords to the concatenated final DataArray
        hist_darr = hist_darr.assign_coords(plot_axis=["l_e", "r_e", "y"])

        restructured_hist_dict[var_name] = hist_darr

    # converting to Dataset
    restructured_hist_ds = xr.Dataset(restructured_hist_dict)
    return restructured_hist_ds
