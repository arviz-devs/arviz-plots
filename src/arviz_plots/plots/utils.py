"""Utilities for batteries included plots."""
from copy import copy
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import references_to_dataset
from arviz_base.utils import _var_names

from arviz_plots.plot_collection import concat_model_dict, process_facet_dims
from arviz_plots.visuals import hline, hspan, vline, vspan


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


def filter_aes(pc, aes_by_visuals, artist, sample_dims):
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
    artist_aes = aes_by_visuals.get(artist, {})
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
    figsize = pc_kwargs["figure_kwargs"].get("figsize", None)
    figsize_units = pc_kwargs["figure_kwargs"].get("figsize_units", "inches")
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

    pc_kwargs["figure_kwargs"]["figsize"] = figsize
    pc_kwargs["figure_kwargs"]["figsize_units"] = figsize_units
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
    figsize = pc_kwargs["figure_kwargs"].get("figsize", None)
    figsize_units = pc_kwargs["figure_kwargs"].get("figsize_units", "inches")
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

    pc_kwargs["figure_kwargs"]["figsize"] = figsize
    pc_kwargs["figure_kwargs"]["figsize_units"] = figsize_units
    return pc_kwargs


def add_lines(
    plot_collection,
    values,
    orientation="vertical",
    aes_by_visuals=None,
    visuals=None,
    sample_dims=None,
    ref_dim="ref_dim",
    **kwargs,
):
    """Add lines.

    This function adds lines to a plot collection based on the provided values.
    It supports both vertical and horizontal lines, depending on the specified orientation.

    Parameters
    ----------
    plot_collection : PlotCollection
        Plot collection to which the lines will be added.
    values : int, float, tuple, list or dict
        Positions for the lines.
    orientation : str, default "vertical"
        The orientation of the lines, either "vertical" or "horizontal".
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

        The default is to use an "overlay_ref" aesthetic for all elements.

        It is possible to request aesthetics without mappings defined in the
        provided `plot_collection`. In those cases, a mapping of "ref_dim" to the requested
        aesthetic will be automatically added.
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * "ref_line" -> passed to :func:`~arviz_plots.visuals.vline` for vertical `orientation`
          and to :func:`~arviz_plots.visuals.hline` for horizontal `orientation`
        * "ref_text" -> TODO

    sample_dims : list, optional
        Dimensions that should not be added to the Dataset generated from
        `values` via :func:`arviz_base.references_to_dataset`.
        Defaults to all dimensions in ``plot_collection.data`` that are not ``facet_dims``
    ref_dim : str, optional
        Specifies the name of the dimension for the line values.
        Defaults to "ref_dim".
    **kwargs : mapping of {str : sequence}, optional
        Mapping of aesthetic keys to the values to be used in their mapping.
        See :func:`~arviz_plots.PlotCollection.generate_aes_dt` for more details.

    Returns
    -------
    plot_collection : PlotCollection
        Plot collection with the lines added.

    Examples
    --------
    Add lines at values 0 and 5 for all variables.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_dist, add_lines, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> pc = plot_dist(
        >>>     dt,
        >>>     kind="ecdf",
        >>>     var_names=["mu"],
        >>> )
        >>> add_lines(pc, values=[0, 5])
    """
    if visuals is None:
        visuals = {}
    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals["ref_line"] = aes_by_visuals.get("ref_line", ["overlay_ref"])
    aes_by_visuals["ref_text"] = aes_by_visuals.get("ref_text", ["overlay_ref"])
    if sample_dims is None:
        sample_dims = list(set(plot_collection.data.dims).difference(plot_collection.facet_dims))
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    plot_bknd = import_module(f".backend.{plot_collection.backend}", package="arviz_plots")

    plot_func = vline if orientation == "vertical" else hline

    ref_ds = references_to_dataset(
        values, plot_collection.data, sample_dims=sample_dims, ref_dim=ref_dim
    )
    requested_aes = (
        set(aes_by_visuals["ref_line"])
        .union(aes_by_visuals["ref_text"])
        .difference(plot_collection.aes_set)
    )
    if ref_dim in ref_ds.dims:
        for aes_key in requested_aes:
            aes_values = np.array(plot_bknd.get_default_aes(aes_key, ref_ds.sizes[ref_dim], kwargs))
            plot_collection.update_aes_from_dataset(
                aes_key,
                xr.Dataset(
                    {
                        var_name: (ref_dim, aes_values)
                        for var_name in plot_collection.data.data_vars
                    },
                    coords={ref_dim: ref_ds[ref_dim]},
                ),
            )

    _, ref_aes, ref_ignore = filter_aes(plot_collection, aes_by_visuals, "ref_line", sample_dims)
    ref_kwargs = copy(visuals.get("ref_line", {}))
    if ref_kwargs is not False:
        if "color" not in ref_aes:
            ref_kwargs.setdefault("color", "gray")
        if "linestyle" not in ref_aes:
            ref_kwargs.setdefault("linestyle", plot_bknd.get_default_aes("linestyle", 2)[1])

        plot_collection.map(
            plot_func,
            "ref_line",
            data=ref_ds,
            ignore_aes=ref_ignore,
            **ref_kwargs,
        )
    return plot_collection


def add_bands(
    plot_collection,
    values,
    orientation="vertical",
    aes_by_visuals=None,
    visuals=None,
    sample_dims=None,
    ref_dim=None,
    **kwargs,
):
    """Add bands.

    This function adds bands (shared areas) to a plot collection based on the provided
    values. It supports both vertical and horizontal bands, depending on the
    specified orientation.

    Parameters
    ----------
    plot_collection : PlotCollection
        Plot collection to which the bands will be added.
    values : tuple, list or dict
        Start and end values for the bands to be plotted.
    orientation : str, default "vertical"
        The orientation of the bands, either "vertical" or "horizontal".
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

        The default is to use an "overlay_band" aesthetic for all elements.

        It is possible to request aesthetics without mappings defined in the
        provided `plot_collection`. In those cases, a mapping of the dimensions in
        `ref_dim` minus its last element to the requested aesthetic will be
        automatically added.
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * "ref_band" -> passed to :func:`~arviz_plots.visuals.vspan` for vertical `orientation`
          and to :func:`~arviz_plots.visuals.hspan` for horizontal `orientation`

    sample_dims : list, optional
        Dimensions that should not be added to the Dataset generated from
        `values` via :func:`arviz_base.references_to_dataset`.
        Defaults to all dimensions in ``plot_collection.data`` that are not ``facet_dims``
    ref_dim : list, optional
        List of dimension names that define the axes along which the band values are stored.
        These dimensions are used to align or compare input data with band data.
        Defaults to ["ref_dim", "band_dim"].
    **kwargs : sequence, optional
        Mapping of aesthetic keys to the values to be used in their mapping.
        See :func:`~arviz_plots.PlotCollection.generate_aes_dt` for more details.

    Returns
    -------
    plot_collection : PlotCollection
        Plot collection with the bands added.

    Examples
    --------
    Add two bands for the theta variable, one from -2 to 2 and the other from -5 to 5.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_dist, add_bands, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> pc = plot_dist(dt)
        >>> add_bands(pc, values=[[-2, 2], [-5, 5]])
    """
    if visuals is None:
        visuals = {}
    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals["ref_band"] = aes_by_visuals.get("ref_band", ["overlay_band"])
    if sample_dims is None:
        sample_dims = list(set(plot_collection.data.dims).difference(plot_collection.facet_dims))
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if ref_dim is None:
        ref_dim = ["ref_dim", "band_dim"]

    plot_func = vspan if orientation == "vertical" else hspan

    ref_ds = references_to_dataset(
        values, plot_collection.data, sample_dims=sample_dims, ref_dim=ref_dim
    )

    requested_aes = set(aes_by_visuals["ref_band"]).difference(plot_collection.aes_set)
    *ref_dim, band_dim = ref_dim
    if ref_ds.sizes[band_dim] != 2:
        raise ValueError(
            f"Expected dimension '{band_dim}' in reference dataset to have size 2, "
            f"but found size {ref_ds.sizes[band_dim]}"
        )
    aes_dt = plot_collection.generate_aes_dt(
        {aes: ref_dim for aes in requested_aes}, ref_ds, **kwargs
    )
    for aes, child in aes_dt.children.items():
        plot_collection.update_aes_from_dataset(aes, child.dataset)

    _, ref_aes, ref_ignore = filter_aes(plot_collection, aes_by_visuals, "ref_band", sample_dims)
    ref_kwargs = copy(visuals.get("ref_band", {}))
    if ref_kwargs is not False:
        if "color" not in ref_aes:
            ref_kwargs.setdefault("color", "gray")
        if "alpha" not in ref_aes:
            ref_kwargs.setdefault("alpha", 0.25)
        plot_collection.map(plot_func, "ref_band", data=ref_ds, ignore_aes=ref_ignore, **ref_kwargs)

    return plot_collection
