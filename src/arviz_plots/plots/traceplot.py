"""Trace plot code."""
from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection, leaf_dataset, process_facet_dims
from arviz_plots.plots.utils import filter_aes, get_group, process_group_variables_coords
from arviz_plots.visuals import labelled_title, labelled_x, line, ticklabel_props, trace_rug


def plot_trace(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
):
    """Plot iteration versus sampled values.

    Parameters
    ----------
    dt : DataTree
        Input data
    var_names: str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars: {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Defaults to only mapping properties to the trace lines.
    plot_kwargs : mapping, optional
        Valid keys are:

        * trace -> passed to :func:`~.visuals.line`
        * divergence -> passed to :func:`~.visuals.trace_rug`
        * title -> :func:`~.visuals.labelled_title`
        * xlabel -> :func:`~.visuals.labelled_x`

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection

    Examples
    --------
    The following examples focus on behaviour specific to ``plot_trace``.
    For a general introduction to batteries-included functions like this one and common
    usage examples see :ref:`plots_intro`

    Default plot_trace

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_trace, style
        >>> style.use("arviz-clean")
        >>> from arviz_base import load_arviz_data
        >>> centered = load_arviz_data('centered_eight')
        >>> plot_trace(centered)

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        figsize = pc_kwargs.get("plot_grid_kws", {}).get("figsize", None)
        figsize_units = pc_kwargs.get("plot_grid_kws", {}).get("figsize_units", "inches")
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in distribution.dims if dim not in sample_dims]
        )
        n_plots, _ = process_facet_dims(distribution, pc_kwargs["cols"])
        col_wrap = pc_kwargs["col_wrap"]
        if n_plots <= col_wrap:
            n_rows, n_cols = 1, n_plots
        else:
            div_mod = divmod(n_plots, col_wrap)
            n_rows = div_mod[0] + (div_mod[1] != 0)
            n_cols = col_wrap
    else:
        figsize, figsize_units = plot_bknd.get_figsize(plot_collection)
        n_rows = leaf_dataset(plot_collection.viz, "row").max().to_array().max()
        n_cols = leaf_dataset(plot_collection.viz, "col").max().to_array().max()

    figsize, textsize, linewidth = plot_bknd.scale_fig_size(
        figsize,
        rows=n_rows,
        cols=n_cols,
        figsize_units=figsize_units,
    )

    if plot_collection is None:
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("color", ["chain"])
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        if "figsize" not in pc_kwargs["plot_grid_kws"]:
            pc_kwargs["plot_grid_kws"]["figsize"] = figsize
            pc_kwargs["plot_grid_kws"]["figsize_units"] = "dots"
        aux_dim_list = [dim for dim in pc_kwargs["cols"] if dim != "__variable__"]
        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )
    else:
        aux_dim_list = list(
            set(
                dim for child in plot_collection.viz.children.values() for dim in child["plot"].dims
            )
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("trace", plot_collection.aes_set)
    aes_map.setdefault("divergence", {"overlay"})

    if labeller is None:
        labeller = BaseLabeller()

    # trace
    trace_kwargs = copy(plot_kwargs.get("trace", {}))
    if trace_kwargs is False:
        xname = None
    else:
        default_xname = sample_dims[0] if len(sample_dims) == 1 else "draw"
        if (default_xname not in distribution.dims) or (
            not np.issubdtype(distribution[default_xname].dtype, np.number)
        ):
            default_xname = None
        xname = trace_kwargs.get("xname", default_xname)
        trace_kwargs["xname"] = xname
        _, trace_aes, trace_ignore = filter_aes(plot_collection, aes_map, "trace", sample_dims)
        if "width" not in trace_aes:
            trace_kwargs.setdefault("width", linewidth)
        plot_collection.map(
            line,
            "trace",
            data=distribution,
            ignore_aes=trace_ignore,
            **trace_kwargs,
        )

    # divergences
    sample_stats = get_group(dt, "sample_stats", allow_missing=True)
    divergence_kwargs = copy(plot_kwargs.get("divergence", {}))
    if (
        sample_stats is not None
        and "diverging" in sample_stats.data_vars
        and np.any(sample_stats.diverging)
        and divergence_kwargs is not False
    ):
        divergence_mask = dt.sample_stats.diverging
        _, div_aes, div_ignore = filter_aes(plot_collection, aes_map, "divergence", sample_dims)
        if "color" not in div_aes:
            divergence_kwargs.setdefault("color", "black")
        if "marker" not in div_aes:
            divergence_kwargs.setdefault("marker", "|")
        if "width" not in div_aes:
            divergence_kwargs.setdefault("width", linewidth)
        if "size" not in div_aes:
            divergence_kwargs.setdefault("size", 30)
        div_reduce_dims = [dim for dim in distribution.dims if dim not in aux_dim_list]

        plot_collection.map(
            trace_rug,
            "divergence",
            data=distribution,
            ignore_aes=div_ignore,
            xname=xname,
            y=distribution.min(div_reduce_dims),
            mask=divergence_mask,
            **divergence_kwargs,
        )

    # aesthetics
    title_kwargs = copy(plot_kwargs.get("title", {}))
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

    # Add "Steps" as x_label for trace
    xlabel_kwargs = copy(plot_kwargs.get("xlabel", {}))
    if xlabel_kwargs is not False:
        _, xlabel_aes, xlabel_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)

        if "color" not in xlabel_aes:
            xlabel_kwargs.setdefault("color", "black")
        if "size" not in xlabel_aes:
            xlabel_kwargs.setdefault("size", textsize)

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=xlabel_ignore,
            store_artist=False,
            text="Steps" if xname is None else xname.capitalize(),
            **xlabel_kwargs,
        )

    # Adjust tick labels
    ticklabels_kwargs = copy(plot_kwargs.get("ticklabels", {}))
    if ticklabels_kwargs is not False:
        _, _, ticklabels_ignore = filter_aes(plot_collection, aes_map, "ticklabels", sample_dims)
        ticklabels_kwargs.setdefault("size", textsize)
        plot_collection.map(
            ticklabel_props,
            ignore_aes=ticklabels_ignore,
            axis="both",
            store_artist=False,
            **ticklabels_kwargs,
        )

    return plot_collection
