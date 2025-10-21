"""Utilities for batteries included plots."""
import warnings
from copy import copy
from importlib import import_module

import matplotlib as mpl
import matplotlib.colors as mpl_colors
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
    artist_dims : list
        Dimensions that should be reduced for this visual.
        That is, all dimensions in `sample_dims` that are not
        mapped to any aesthetic.
    artist_aes : iterable
    ignore_aes : set
    """
    artist_aes = aes_by_visuals.get(visual, {})
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


def build_coord_labels(khat_data):
    """Generate string labels for all coordinate combinations in the input data.

    Parameters
    ----------
    khat_data : DataArray
        Input data array whose coordinate combinations will be labeled.

    Returns
    -------
    ndarray of str
        Array of string labels with the same shape as `khat_data`.
    """
    dims = khat_data.dims
    if not dims:
        return np.array(["0"], dtype=object)

    labels = np.empty(khat_data.shape, dtype=object)
    coords = {
        dim: khat_data.coords[dim].values if dim in khat_data.coords else None for dim in dims
    }

    for index in np.ndindex(khat_data.shape):
        if len(dims) == 1:
            dim = dims[0]
            coord_values = coords[dim]
            if coord_values is not None and coord_values.size:
                value = coord_values[index[0]]
            else:
                value = index[0]
            if hasattr(value, "item"):
                value = value.item()
            labels[index] = str(value)
        else:
            parts = []
            for dim, idx in zip(dims, index):
                coord_values = coords[dim]
                value = coord_values[idx] if coord_values is not None else idx
                if hasattr(value, "item"):
                    value = value.item()
                parts.append(f"{dim}={value}")
            labels[index] = ", ".join(str(part) for part in parts)

    return labels


def alpha_scaled_colors(base_color, khat_values, good_k_threshold):
    """Create RGBA color array with alpha values scaled by Pareto k diagnostic thresholds.

    Parameters
    ----------
    base_color : str or color-like, optional
        Base color to use for all points. Defaults to "C0" if None.
    khat_values : array_like
        Array of Pareto k diagnostic values used to determine alpha scaling.
    good_k_threshold : float
        Threshold value below which k diagnostics are considered good.

    Returns
    -------
    ndarray
        RGBA array with shape (*khat_values.shape, 4) where alpha channel
        is scaled based on k value thresholds.
    """
    values = np.asarray(khat_values)
    base = base_color or "C0"
    rgba = np.array(mpl_colors.to_rgba(base))
    rgba = np.broadcast_to(rgba, values.shape + (4,)).copy()

    if values.size:
        alphas = 0.5 + 0.2 * (values > good_k_threshold) + 0.3 * (values > 1.0)
        rgba[..., 3] = np.clip(alphas, 0.0, 1.0)
    return rgba


def calculate_khat_bin_edges(values, thresholds, tolerance=1e-9):
    """Calculate bin edges for Pareto k diagnostic bins.

    Parameters
    ----------
    values : array_like
        Pareto k values to bin
    thresholds : sequence of float
        Diagnostic threshold values to use as potential bin edges (e.g., [0.7, 1.0])
    tolerance : float, default 1e-9
        Numerical tolerance for edge comparisons to avoid duplicate edges

    Returns
    -------
    bin_edges : list of float or None
        Calculated bin edges suitable for np.histogram, or None if edges cannot
        be computed.
    """
    if not values.size:
        return None

    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))

    if not (np.isfinite(ymin) and np.isfinite(ymax)):
        return None

    bin_edges = [ymin]

    for edge in thresholds:
        if edge is None or not np.isfinite(edge):
            continue
        if edge <= bin_edges[-1] + tolerance:
            continue
        if edge >= ymax - tolerance:
            continue
        bin_edges.append(float(edge))

    if ymax > bin_edges[-1] + tolerance:
        bin_edges.append(ymax)
    else:
        bin_edges[-1] = ymax
    return bin_edges if len(bin_edges) > 1 else None


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
    labels : array_like
        Array of labels corresponding to each data point.
    colors : array_like or None
        Array of colors for each data point. If None, extracted from scatter plot.
    values : array_like or None
        Array of values to display in hover text. If None, y-coordinates are used.
    """
    if backend != "matplotlib":
        return

    try:
        fig = plot_collection.viz["figure"].item()
    except KeyError:
        return
    if fig is None:
        return

    if hasattr(mpl, "backends") and hasattr(mpl.backends, "backend_registry"):
        interactive_backends = mpl.backends.backend_registry.list_builtin(
            mpl.backends.BackendFilter.INTERACTIVE
        )
    else:
        interactive_backends = mpl.rcsetup.interactive_bk

    if mpl.get_backend() not in interactive_backends:
        warnings.warn(
            "hover labels are only available with interactive backends. "
            "To switch to an interactive backend from IPython or Jupyter, use `%matplotlib`.",
            UserWarning,
        )
        return

    try:
        axis = plot_collection.viz["plot"]["pareto_k"].item()
        scatter = plot_collection.viz["khat"]["pareto_k"].item()
    except KeyError:
        return

    if axis is None or scatter is None:
        return

    color_arr = np.asarray(colors) if colors is not None else None
    if color_arr is None:
        facecolors = scatter.get_facecolors()
        if facecolors.size:
            color_arr = facecolors

    values_arr = np.asarray(values) if values is not None else None

    hover_labels(
        fig,
        axis,
        scatter,
        labels,
        hover_format,
        color_arr,
        values_arr,
    )


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
