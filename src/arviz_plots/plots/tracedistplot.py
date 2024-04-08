"""TraceDist plot code."""
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection, process_facet_dims
from arviz_plots.plots.traceplot import plot_trace
from arviz_plots.plots.utils import filter_aes, get_group, process_group_variables_coords
from arviz_plots.visuals import (
    ecdf_line,
    labelled_x,
    labelled_y,
    line_xy,
    remove_ticks,
    ticklabel_props,
    trace_rug,
)


def plot_trace_dist(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    compact=True,
    combined=False,
    kind=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot 1D marginal distributions and iteration versus sampled values.

    Parameters
    ----------
    dt : DataTree
        Input data
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    compact : bool, default True
        Plot multidimensional variables in a single :term:`plot`.
    combined : bool, default False
        Whether to plot intervals for each chain or not. Ignored when the "chain" dimension
        is not present.
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal distribution.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection` when
        plotted. The defaults depend on the combination of `compact` and `combined`,
        see the examples section for an illustrated description.
        Valid keys are the same as for `plot_kwargs`.
    plot_kwargs : mapping, optional
        Valid keys are:

        * One of "kde", "ecdf", "dot" or "hist", matching the `kind` argument.

          * "kde" -> :func:`~.visuals.line_xy`
          * "ecdf" -> :func:`~.visuals.ecdf_line`

        * "trace" -> passed to :func:`~.visuals.line`
        * "divergence" -> passed to :func:`~.visuals.trace_rug`
        * "label" -> :func:`~.visuals.labelled_x` and :func:`~.visuals.labelled_y`
        * "ticklabels" -> :func:`~.visuals.ticklabel_props`
        * "xlabel_trace" -> :func:`~.visuals.labelled_x`

    stats_kwargs : mapping, optional
        Valid keys are:

        * density -> passed to kde, ecdf, ...

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection

    Examples
    --------
    The following examples focus on behaviour specific to ``plot_trace_dist``.
    For a general introduction to batteries-included functions like this one and common
    usage examples see :ref:`plots_intro`

    Default plot_trace_dist (``compact=True`` and ``combined=False``). In this case,
    the multiple coordinate values are overlaid on the same plot for multidimensional values;
    by default, the color is mapped to all dimensions of each variable (but `sample_dims`)
    to allow distinguising the different coordinate values.

    As ``combined=False`` each chain is also being plotted, overlaying them on their
    corresponding plots; as the color property is already taken, the chain information
    is encoded in the linestyle as default.

    Both mappings are applied to the trace and dist elements.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_trace_dist, style
        >>> style.use("arviz-clean")
        >>> from arviz_base import load_arviz_data
        >>> centered = load_arviz_data('centered_eight')
        >>> coords = {"school": ["Choate", "Deerfield", "Hotchkiss"]}
        >>> pc = plot_trace_dist(centered, coords=coords, compact=True, combined=False)
        >>> pc.add_legend("school")

    plot_trace_dist with ``compact=True`` and ``combined=True``. The aesthetic mappings
    stay the same as in the previous case, but now the linestyle property mapping
    is only taken into account for the trace as in the left column, we use
    the data from all chains to generate a single distribution representation
    for each variable+coordinate value combination.

    Similarly to the first case, this default and now only mapping is applied to both
    the trace and the dist elements.

    .. plot::
        :context: close-figs

        >>> pc = plot_trace_dist(centered, coords=coords, compact=True, combined=True)
        >>> pc.add_legend("school")

    When ``compact=False``, each variable and coordinate value gets its own plot,
    and so the color property is no longer used to encode this information.
    Instead, it is now used to encode the chain information.

    .. plot::
        :context: close-figs

        >>> pc = plot_trace_dist(centered, coords=coords, compact=False, combined=False)

    Similarly to the other ``combined=True`` case, the aesthetics stay the same
    as with ``combined=False``, but they are ignored by default when plotting
    on the left column.

    .. plot::
        :context: close-figs

        >>> pc = plot_trace_dist(centered, coords=coords, compact=False, combined=True)
        >>> pc.add_legend("chain")

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if stats_kwargs is None:
        stats_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    if not combined and "chain" not in distribution.dims:
        combined = True

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        figsize = pc_kwargs.get("plot_grid_kws", {}).get("figsize", None)
        figsize_units = pc_kwargs.get("plot_grid_kws", {}).get("figsize_units", "inches")
        aux_dim_list = [dim for dim in distribution.dims if dim not in sample_dims]
        if compact:
            pc_kwargs["rows"] = ["__variable__"]
        else:
            pc_kwargs.setdefault("rows", ["__variable__"] + aux_dim_list)
            aux_dim_list = [dim for dim in pc_kwargs["rows"] if dim != "__variable__"]
        row_dims = pc_kwargs["rows"]
    else:
        figsize, figsize_units = plot_bknd.get_figsize(plot_collection)
        aux_dim_list = list(
            set(
                dim for child in plot_collection.viz.children.values() for dim in child["plot"].dims
            ).difference({"column"})
        )
        row_dims = ["__variable__"] + aux_dim_list

    figsize, textsize, linewidth = plot_bknd.scale_fig_size(
        figsize,
        rows=process_facet_dims(distribution, row_dims)[0],
        cols=2,
        figsize_units=figsize_units,
    )

    color_cycle = pc_kwargs.get("color", plot_bknd.get_default_aes("color", 10, {}))
    if len(color_cycle) <= 2:
        raise ValueError(
            f"Not enough values provided for color cycle, got {color_cycle} "
            "but at least 3 are needed"
        )
    linestyle_cycle = pc_kwargs.get("linestyle", plot_bknd.get_default_aes("linestyle", 4, {}))
    if len(linestyle_cycle) <= 2:
        raise ValueError(
            f"Not enough values provided for linestyle cycle, got {linestyle_cycle} "
            "but at least 3 are needed"
        )
    if not compact and combined:
        neutral_color = color_cycle[0]
        pc_kwargs["color"] = color_cycle[1:]
    else:
        neutral_color = False

    if compact and combined:
        neutral_linestyle = linestyle_cycle[0]
        pc_kwargs["linestyle"] = linestyle_cycle[1:]
    else:
        neutral_linestyle = False

    if plot_collection is None:
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        if "figsize" not in pc_kwargs["plot_grid_kws"]:
            pc_kwargs["plot_grid_kws"]["figsize"] = figsize
            pc_kwargs["plot_grid_kws"]["figsize_units"] = "dots"
        if compact:
            pc_kwargs["aes"].setdefault("color", ["__variable__"] + aux_dim_list)
            if "chain" in distribution.dims:
                pc_kwargs["aes"].setdefault("overlay", ["__variable__", "chain"] + aux_dim_list)
                pc_kwargs["aes"].setdefault("linestyle", ["chain"])
            else:
                pc_kwargs["aes"].setdefault("overlay", ["__variable__"] + aux_dim_list)
        elif "chain" in distribution.dims:
            pc_kwargs["aes"].setdefault("color", ["chain"])
            pc_kwargs["aes"].setdefault("overlay", ["chain"])

        plot_collection = PlotCollection.grid(
            distribution.expand_dims(column=2).assign_coords(column=["dist", "trace"]),
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    if combined and "chain" in distribution.dims:
        if compact:
            aes_map[kind] = aes_map.get(
                kind, plot_collection.aes_set.difference({"overlay", "linestyle"})
            )
        else:
            aes_map[kind] = aes_map.get(
                kind, plot_collection.aes_set.difference({"overlay", "color"})
            )
    else:
        aes_map[kind] = {"overlay"}.union(aes_map.get(kind, plot_collection.aes_set))
    aes_map["trace"] = {"overlay"}.union(aes_map.get("trace", plot_collection.aes_set))
    aes_map["divergence"] = {"overlay"}.union(aes_map.get("divergence", {}))

    if combined and "chain" in distribution.dims:
        chain_mapped_to_aes = set(
            aes_key for aes_key, aes_dims in pc_kwargs["aes"].items() if "chain" in aes_dims
        ).intersection(aes_map[kind])
        if chain_mapped_to_aes:
            raise ValueError(
                f"Found properties {chain_mapped_to_aes} mapped to the chain dimension, "
                "but combined=True. Set combined=False or modify the aesthetic mappings"
            )

    if labeller is None:
        labeller = BaseLabeller()

    dist_dims, dist_aes, dist_ignore = filter_aes(plot_collection, aes_map, kind, sample_dims)

    # dens
    dist_kwargs = plot_kwargs.get(kind, {}).copy()
    if "linewidth" not in dist_aes:
        dist_kwargs.setdefault("width", linewidth)
    if neutral_color and "color" not in dist_aes:
        dist_kwargs.setdefault("color", neutral_color)
    if neutral_linestyle and "linestyle" not in dist_aes:
        dist_kwargs.setdefault("linestyle", neutral_linestyle)
    if kind == "kde":
        density = distribution.azstats.kde(dims=dist_dims, **stats_kwargs.get("density", {}))
        plot_collection.map(
            line_xy,
            "dist",
            data=density,
            ignore_aes=dist_ignore,
            coords={"column": "dist"},
            **dist_kwargs,
        )

    elif kind == "ecdf":
        density = distribution.azstats.ecdf(dims=dist_dims, **stats_kwargs.get("density", {}))
        plot_collection.map(
            ecdf_line,
            "dist",
            data=density,
            ignore_aes=dist_ignore,
            coords={"column": "dist"},
            **dist_kwargs,
        )

    # trace
    plot_kwargs_trace = {
        key.replace("_trace", ""): value
        for key, value in plot_kwargs.items()
        if key in {"trace", "divergence", "xlabel_trace"}
    }
    plot_kwargs_trace["title"] = False
    plot_kwargs_trace["ticklabels"] = False
    aes_map_trace = {
        key.replace("_trace", ""): value
        for key, value in plot_kwargs.items()
        if key in {"trace", "divergence", "xlabel_trace"}
    }
    plot_collection.coords = {"column": "trace"}
    plot_trace(
        dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_map=aes_map_trace,
        plot_kwargs=plot_kwargs_trace,
    )
    plot_collection.coords = None
    plot_collection.rename_artists(xlabel="xlabel_trace", divergence="divergence_trace")
    # divergences
    sample_stats = get_group(dt, "sample_stats", allow_missing=True)
    if (
        sample_stats is not None
        and "diverging" in sample_stats.data_vars
        and np.any(sample_stats.diverging)
    ):
        divergence_mask = dt.sample_stats.diverging
        _, div_aes, div_ignore = filter_aes(plot_collection, aes_map, "divergence", sample_dims)
        divergence_kwargs = plot_kwargs.get("divergence", {}).copy()
        if "color" not in div_aes:
            divergence_kwargs.setdefault("color", "black")
        if "marker" not in div_aes:
            divergence_kwargs.setdefault("marker", "|")
        if "width" not in div_aes:
            divergence_kwargs.setdefault("width", linewidth)
        if "size" not in div_aes:
            divergence_kwargs.setdefault("size", 30)

        plot_collection.map(
            trace_rug,
            "divergence_dist",
            data=distribution,
            ignore_aes=div_ignore,
            xname=False,
            y=0,
            coords={"column": "dist"},
            mask=divergence_mask,
            **divergence_kwargs,
        )

    ## aesthetics
    # Remove yticks, only for KDEs
    if kind == "kde":
        _, _, yticks_dist_ignore = filter_aes(plot_collection, aes_map, "yticks_dist", sample_dims)

        plot_collection.map(
            remove_ticks,
            "yticks_dist",
            ignore_aes=yticks_dist_ignore,
            coords={"column": "dist"},
            store_artist=False,
            axis="y",
        )

    # Add varnames as x and y labels
    _, labels_aes, labels_ignore = filter_aes(plot_collection, aes_map, "label", sample_dims)
    label_kwargs = plot_kwargs.get("label", {}).copy()

    if "color" not in labels_aes:
        label_kwargs.setdefault("color", "black")

    label_kwargs.setdefault("size", textsize)

    plot_collection.map(
        labelled_x,
        "xlabel_dist",
        ignore_aes=labels_ignore,
        coords={"column": "dist"},
        subset_info=True,
        labeller=labeller,
        store_artist=False,
        **label_kwargs,
    )

    plot_collection.map(
        labelled_y,
        "ylabel_trace",
        ignore_aes=labels_ignore,
        coords={"column": "trace"},
        subset_info=True,
        labeller=labeller,
        store_artist=False,
        **label_kwargs,
    )

    # Adjust tick labels
    ticklabels_kwargs = plot_kwargs.get("ticklabels", {}).copy()
    ticklabels_kwargs.setdefault("size", textsize)
    _, _, ticklabels_ignore = filter_aes(plot_collection, aes_map, "ticklabels", sample_dims)
    plot_collection.map(
        ticklabel_props,
        ignore_aes=ticklabels_ignore,
        axis="both",
        store_artist=False,
        **ticklabels_kwargs,
    )

    return plot_collection
