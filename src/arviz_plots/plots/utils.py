"""Utilities for batteries included plots."""
import warnings
from copy import copy
from importlib import import_module

import numpy as np
import xarray as xr
from arviz_base import rcParams, references_to_dataset, xarray_sel_iter
from arviz_base.labels import BaseLabeller
from arviz_base.utils import _var_names
from arviz_stats import ecdf, histogram, kde

from arviz_plots.plot_collection import concat_model_dict, process_facet_dims
from arviz_plots.visuals import annotate_xy, hline, hspan, vline, vspan


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


def get_visual_kwargs(visuals, name, default=None):
    """Get the kwargs for a visual from the visuals dict.

    Parameterss
    ----------
    visuals : dict
        Dictionary of visuals.
    name : str
        Name of the visual to get kwargs for.
    default : dict or False, optional
        Default kwargs to return if the visual is not found in `visuals`.

    Returns
    -------
    dict
        The kwargs for the visual if found, otherwise the default value.

    """
    if default is None:
        default = {}
    vis_kwargs = copy(visuals.get(name, default))
    if vis_kwargs is True:
        vis_kwargs = {}
    return vis_kwargs


def process_group_variables_coords(dt, group, var_names, filter_vars, coords, allow_dict=True):
    """Process main input arguments of batteries included plotting functions."""
    if coords is None:
        coords = {}
    if isinstance(dt, dict) and not allow_dict:
        raise ValueError("Input data as dictionary not supported")
    if isinstance(dt, dict):
        distribution = {}
        all_vars = []
        all_data_vars = []
        for key, value in dt.items():
            new_var_names = _var_names(
                var_names, get_group(value, group), filter_vars, check_if_present=False
            )
            group_ds = get_group(value, group)
            if new_var_names is not None:
                data_vars = group_ds.data_vars
                available_vars = [v for v in data_vars if v in new_var_names]
                distribution[key] = group_ds[available_vars].sel(coords)
                all_vars.extend(new_var_names)
                all_data_vars.extend(data_vars)
            else:
                distribution[key] = group_ds.sel(coords)

        if var_names is not None:
            missing_vars = set(all_vars).difference(set(all_data_vars))
            if missing_vars:
                plural = "" if len(missing_vars) == 1 else "s"
                raise KeyError(f"variable{plural} {missing_vars} not found in any dataset")

        distribution = concat_model_dict(distribution)
    else:
        distribution = get_group(dt, group)
        var_names = _var_names(var_names, distribution, filter_vars)
        if var_names is not None:
            distribution = distribution[var_names]
        distribution = distribution.sel(coords)
    return distribution


def filter_aes(pc, aes_by_visuals, visual, sample_dims):
    """Split aesthetics and get relevant dimensions.

    Returns
    -------
    reduce_dims : list
        Dimensions that should be reduced for this visual.
        That is, all dimensions in `sample_dims` that are not
        mapped to any aesthetic.
    active_dims : list
        Dimensions that have either faceting or aesthetic mappings
        active for that visual. Should not be reduced and should have
        a groupby performed on them if computing summaries.
    artist_aes : iterable
    ignore_aes : set
    """
    artist_aes = aes_by_visuals.get(visual, {})
    pc_aes = pc.aes_set
    ignore_aes = set(pc_aes).difference(artist_aes)
    _, all_loop_dims = pc.update_aes(ignore_aes=ignore_aes)
    reduce_dims = [dim for dim in sample_dims if dim not in all_loop_dims]
    active_dims = [dim for dim in all_loop_dims if dim not in sample_dims]
    return reduce_dims, active_dims, artist_aes, ignore_aes


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

def compute_dist(data, reduce_dims, active_dims, kind=None, stats=None):
    if stats is None:
        stats = {}
    # quick exit if pre-computed elements in `stats`
    if any(isinstance(stats.get(viz, None), xr.Dataset) for viz in ("ecdf", "hist", "kde")):
        return (stats.get(viz, xr.Dataset()) for viz in ("ecdf", "hist", "kde"))
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if set(reduce_dims).intersection(active_dims):
        raise ValueError("'reduce_dims' and 'active_dims' can't share elements")
    ecdf_vars = []
    hist_vars = []
    kde_vars = []
    if kind == "auto":
        for var_name, da in data.items():
            reduced_size = np.prod([da.sizes[dim] for dim in reduce_dims if dim in da.dims]) 
            groupby_dims = [dim for dim in active_dims if dim in da.dims]
            if groupby_dims:
                reduced_size *= np.prod([np.min(np.unique(da.coords[dim], return_counts=True)[1]) for dim in groupby_dims])
            print(f"{var_name=}, {reduced_size=}")
            if reduced_size < 100:
                ecdf_vars.append(var_name)
            elif da.dtype.kind == "f":
                kde_vars.append(var_name)
            else:
                hist_vars.append(var_name)
    elif kind == "ecdf":
        ecdf_vars == list(data.data_vars)
    elif kind == "hist":
        hist_vars == list(data.data_vars)
    elif kind == "kde":
        kde_vars = list(data.data_vars)
    
    if ecdf_vars:
        ecdf_data = data[ecdf_vars]
        groupby_dims = [dim for dim in active_dims if dim in ecdf_data.dims]
        if groupby_dims:
            ecdf_data = ecdf_data.groupby(groupby_dims)
        ecdf_out = ecdf(ecdf_data, dim=reduce_dims, **stats.get("ecdf", {}))
    else:
        ecdf_out = xr.Dataset()

    if hist_vars:
        hist_data = data[hist_vars]
        groupby_dims = [dim for dim in active_dims if dim in hist_data.dims]
        if groupby_dims:
            hist_data = hist_data.groupby(groupby_dims)
        hist_out = histogram(hist_data, dim=reduce_dims, **stats.get("hist", {}))
    else:
        hist_out = xr.Dataset()

    if kde_vars:
        kde_data = data[kde_vars]
        groupby_dims = [dim for dim in active_dims if dim in kde_data.dims]
        if groupby_dims:
            kde_data = kde_data.groupby(groupby_dims)
        kde_out = kde(kde_data, dim=reduce_dims, **stats.get("kde", {}))
    else:
        kde_out = xr.Dataset()

    return ecdf_out, hist_out, kde_out


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
    visuals : mapping of {str : mapping or bool}, optional
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
    ref_kwargs = get_visual_kwargs(visuals, "ref_line")
    if ref_kwargs is not False:
        if "color" not in ref_aes:
            ref_kwargs.setdefault("color", "B2")
        if "linestyle" not in ref_aes:
            ref_kwargs.setdefault("linestyle", "C1")

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
    visuals : mapping of {str : mapping or bool}, optional
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
    ref_kwargs = get_visual_kwargs(visuals, "ref_band")
    if ref_kwargs is not False:
        if "color" not in ref_aes:
            ref_kwargs.setdefault("color", "B2")
        if "alpha" not in ref_aes:
            ref_kwargs.setdefault("alpha", 0.25)
        plot_collection.map(plot_func, "ref_band", data=ref_ds, ignore_aes=ref_ignore, **ref_kwargs)

    return plot_collection


def format_coords_as_labels(data, skip_dims=None, labeller=None):
    """Format 1D or multi-D dataarray coords as string labels.

    Parameters
    ----------
    data : xr.DataArray
        DataArray whose coordinates will be converted to labels.
    skip_dims : str or list_like, optional
        Dimensions whose values should not be included in the labels.
    labeller : BaseLabeller, optional
        Labeller instance to use for formatting. If None, defaults to BaseLabeller().
        Users can pass different labellers to control whether indices or coordinate
        values are shown (e.g., IdxLabeller for indices).

    Returns
    -------
    ndarray of str
        Array of coordinate labels with the same flattened shape as the input.
    """
    if labeller is None:
        labeller = BaseLabeller()

    if skip_dims is None:
        skip_dims = []
    elif isinstance(skip_dims, str):
        skip_dims = [skip_dims]

    dims_to_include = [dim for dim in data.dims if dim not in skip_dims]

    if not dims_to_include:
        return np.array([], dtype=object)

    shape = [data.sizes[dim] for dim in dims_to_include]
    n_points = int(np.prod(shape))

    index_arrays = [
        arr.ravel() for arr in np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    ]
    coord_arrays = [data.coords[dim].values for dim in dims_to_include]

    labels = [
        labeller.sel_to_str(
            {dim: coord_arrays[j][index_arrays[j][i]] for j, dim in enumerate(dims_to_include)},
            {dim: int(index_arrays[j][i]) for j, dim in enumerate(dims_to_include)},
        )
        for i in range(n_points)
    ]
    return np.array(labels, dtype=object)


def annotate_bin_text(da, target, x, y, count_da, n_da, bin_format, **kwargs):
    """Format and annotate bin text with count and percentage.

    Parameters
    ----------
    da : xr.DataArray
        Data array to annotate
    target : Axes
        Target axes for annotation
    x : float
        X-position for annotation
    y : float
        Y-position for annotation
    count_da : int or xr.DataArray
        Count value for the bin
    n_da : int or xr.DataArray
        Total count value
    bin_format : str
        Format string for bin text (supports {count} and {pct})
    **kwargs
        Additional keyword arguments passed to annotate_xy

    Returns
    -------
    Artist
        The annotation artist
    """
    if hasattr(count_da, "values"):
        arr = count_da.values
        count_val = int(arr[()] if arr.ndim == 0 else arr)
    else:
        count_val = int(count_da)

    if hasattr(n_da, "values"):
        arr = n_da.values
        n_val = int(arr[()] if arr.ndim == 0 else arr)
    else:
        n_val = int(n_da)

    pct = (count_val / n_val * 100) if n_val > 0 else 0.0
    text_str = bin_format.format(count=count_val, pct=pct)
    return annotate_xy(da, target, x=x, y=y, text=text_str, **kwargs)


def enable_hover_labels(backend, plot_collection, hover_format, labels, colors, values):
    """Set up interactive hover annotations for scatter plots on matplotlib backends.

    Parameters
    ----------
    backend : str
        The plotting backend being used. Only "matplotlib" is supported.
    plot_collection : PlotCollection
        Plot collection containing the visualization elements.
    hover_format : str
        Format string template for hover annotation text.
    labels : xr.DataArray
        Labels corresponding to each data point as a DataArray. It is subset per
        facet/aesthetic combination automatically.
    colors : xr.DataArray or None
        Colors for each data point as a DataArray, or None to extract from scatter artist.
    values : xr.DataArray or None
        Values to display in hover text as a DataArray, or None to use y-coordinates.
    """
    if backend != "matplotlib":
        return

    try:
        fig = plot_collection.viz["figure"].item()
    except KeyError:
        return
    if fig is None:
        return

    if not hasattr(fig.canvas, "mpl_connect"):
        warnings.warn(
            "hover labels are only available with interactive backends. "
            "To switch to an interactive backend from IPython or Jupyter, use `%matplotlib`.",
            UserWarning,
        )
        return

    try:
        scatter_tree = plot_collection.viz["khat"]
        scatter_da = scatter_tree["pareto_k"]
    except KeyError:
        return

    if scatter_da.ndim == 0:
        scatter = scatter_da.item()
        axis = plot_collection.get_target("pareto_k", {})

        if scatter is not None and axis is not None:
            label_arr = labels.to_numpy().ravel()
            label_list = [str(label) for label in label_arr]
            value_arr = values.to_numpy().ravel() if values is not None else None
            color_arr = colors.to_numpy().ravel() if colors is not None else None

            hover_labels(fig, axis, scatter, label_list, hover_format, color_arr, value_arr)
    else:
        scatter_ds = scatter_da.to_dataset(name="artist")
        for _, sel, _ in xarray_sel_iter(scatter_ds, skip_dims=set()):
            scatter = scatter_ds["artist"].sel(sel).item()
            if scatter is None:
                continue
            axis = plot_collection.get_target("pareto_k", sel)
            if axis is None:
                continue

            sel_args = {dim: val for dim, val in sel.items() if dim in labels.dims}
            label_subset = labels.sel(sel_args) if sel_args else labels
            label_arr = label_subset.to_numpy().ravel()

            offsets = scatter.get_offsets()
            if offsets is None or not offsets.size:
                continue
            if len(label_arr) != len(offsets):
                continue

            label_list = [str(label) for label in label_arr]

            value_arr = None
            if values is not None:
                sel_args = {dim: val for dim, val in sel.items() if dim in values.dims}
                value_subset = values.sel(sel_args) if sel_args else values
                value_arr = value_subset.to_numpy().ravel()

            color_arr = None
            if colors is not None:
                sel_args = {dim: val for dim, val in sel.items() if dim in colors.dims}
                color_subset = colors.sel(sel_args) if sel_args else colors
                color_arr = color_subset.to_numpy().ravel()

            hover_labels(fig, axis, scatter, label_list, hover_format, color_arr, value_arr)


def hover(
    event, annot, ax, scatter, fig, offsets, labels, values, hover_format, colors, offset_distance
):
    """Handle mouse hover events to show or hide data point annotations.

    Parameters
    ----------
    event : MouseEvent
        Matplotlib mouse event containing cursor position.
    annot : Annotation
        Matplotlib annotation object to show/hide and update.
    ax : Axes
        Matplotlib axes containing the scatter plot.
    scatter : PathCollection
        Scatter plot artist whose points are being tracked.
    fig : Figure
        Matplotlib figure containing the plot.
    offsets : ndarray
        Array of (x, y) coordinates for all scatter plot points.
    labels : array_like
        Array of string labels for each data point.
    values : array_like or None
        Array of numeric values to display. If None, y-coordinates are used.
    hover_format : str
        Format string template for the annotation text.
    colors : ndarray or None
        Array of RGBA colors for annotation backgrounds.
    offset_distance : float
        Distance in points to offset the annotation from the cursor position.
    """
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont:
            idx = ind["ind"][0]
            pos = offsets[idx]
            label = labels[idx] if idx < len(labels) else str(idx)
            value = values[idx] if values is not None and idx < len(values) else pos[1]
            annot.xy = pos
            xmid = np.mean(ax.get_xlim())
            ymid = np.mean(ax.get_ylim())
            annot.set_position(
                (
                    -offset_distance if pos[0] > xmid else offset_distance,
                    -offset_distance if pos[1] > ymid else offset_distance,
                )
            )
            annot.set_text(_format_hover_text(hover_format, idx, label, value))

            if colors is not None and idx < len(colors):
                annot.get_bbox_patch().set_facecolor(colors[idx])

            annot.set_ha("right" if pos[0] > xmid else "left")
            annot.set_va("top" if pos[1] > ymid else "bottom")
            annot.set_visible(True)
            fig.canvas.draw_idle()
        elif vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()


def hover_labels(fig, ax, scatter, labels, hover_format, colors, values):
    """Configure hover annotation display for scatter plot data points.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to attach the hover event handler.
    ax : Axes
        Matplotlib axes containing the scatter plot.
    scatter : PathCollection
        Scatter plot artist whose data points trigger hover annotations.
    labels : array_like
        Array of string labels for each data point.
    hover_format : str
        Format string template for annotation text.
    colors : ndarray or None
        Array of RGBA colors for annotation box backgrounds.
    values : array_like or None
        Array of numeric values to display. If None, y-coordinates are used.
    """
    offsets = scatter.get_offsets()
    if offsets is None or not offsets.size:
        return

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "w", "alpha": 0.4},
        arrowprops={"arrowstyle": "->"},
    )
    annot.set_visible(False)
    offset_distance = 10

    fig.canvas.mpl_connect(
        "motion_notify_event",
        lambda event: hover(
            event,
            annot,
            ax,
            scatter,
            fig,
            offsets,
            labels,
            values,
            hover_format,
            colors,
            offset_distance,
        ),
    )


def _format_hover_text(template, index, label, value):
    """Format hover annotation text using named placeholders."""
    if hasattr(value, "item"):
        value = value.item()
    return template.format(index=index, label=label, value=value)
