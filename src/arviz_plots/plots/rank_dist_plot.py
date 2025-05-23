# pylint: disable=R0801
"""rankDist plot code."""
from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.rank_plot import plot_rank
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    process_group_variables_coords,
    set_grid_layout,
)
from arviz_plots.visuals import labelled_x, labelled_y, ticklabel_props, trace_rug


def plot_rank_dist(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    compact=True,
    combined=False,
    kind=None,
    ci_prob=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_dist_kwargs=None,
    stats_rank_kwargs=None,
    pc_kwargs=None,
):
    """Plot 1D marginal distributions and fractional rank Δ-ECDF plots.

    Rank plots are built by replacing the posterior draws by their ranking computed over all chains.
    Then each chain is plotted independently. If all of the chains are targeting the same posterior,
    we expect the ranks in each chain to be uniformly distributed.
    To simplify comparison we compute the ordered fractional ranks, which are distributed
    uniformly in [0, 1]. Additionally, we plot the Δ-ECDF, that is, the difference between the
    expected CDF from the observed ECDF.
    Simultaneous confidence bands are computed using the method described in [1]_.

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
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval for
        the fractional ranks.
        Defaults to ``rcParams["stats.ci_prob"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection` when
        plotted. The defaults depend on the combination of `compact` and `combined`,
        see the examples section for an illustrated description.
        Valid keys are the same as for `plot_kwargs`.
    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * One of "kde", "ecdf", "dot" or "hist", matching the `kind` argument.

          * "kde" -> :func:`~.visuals.line_xy`
          * "ecdf" -> :func:`~.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * "rank" -> passed to :func:`~.visuals.ecdf_line`
        * "label" -> :func:`~.visuals.labelled_x` and :func:`~.visuals.labelled_y`
        * "ticklabels" -> :func:`~.visuals.ticklabel_props`
        * "xlabel_rank" -> :func:`~.visuals.labelled_x`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats_dist_kwargs : mapping, optional
        Valid keys are:

        * density -> passed to kde, ecdf, ...

    stats_rank_kwargs : mapping, optional
        Valid keys are:

        * n_simulations -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`. Default is 1000.
        * method -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`. Default is "simulation".

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection

    Examples
    --------
    The following examples focus on behaviour specific to ``plot_rank_dist``.
    For a general introduction to batteries-included functions like this one and common
    usage examples see :ref:`plots_intro`

    Default plot_rank_dist (``compact=True`` and ``combined=False``). In this case,
    the multiple coordinate values are overlaid on the same plot for multidimensional values;
    by default, the color is mapped to all dimensions of each variable (but `sample_dims`)
    to allow distinguishing the different coordinate values.

    As ``combined=False`` each chain is also being plotted, overlaying them on their
    corresponding plots; as the color property is already taken, the chain information
    is encoded in the linestyle as default.

    Both mappings are applied to the rank and dist elements.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_rank_dist, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> centered = load_arviz_data('centered_eight')
        >>> coords = {"school": ["Choate", "Deerfield", "Hotchkiss"]}
        >>> pc = plot_rank_dist(centered, coords=coords, compact=True, combined=False)
        >>> pc.add_legend(["__variable__", "school"])

    plot_rank_dist with ``compact=True`` and ``combined=True``. The aesthetic mappings
    stay the same as in the previous case, but now the linestyle property mapping
    is only taken into account for the rank as in the left column, we use
    the data from all chains to generate a single distribution representation
    for each variable+coordinate value combination.

    Similarly to the first case, this default and now only mapping is applied to both
    the rank and the dist elements.

    .. plot::
        :context: close-figs

        >>> pc = plot_rank_dist(centered, coords=coords, compact=True, combined=True)
        >>> pc.add_legend(["__variable__", "school"])

    When ``compact=False``, each variable and coordinate value gets its own plot,
    and so the color property is no longer used to encode this information.
    Instead, it is now used to encode the chain information.

    .. plot::
        :context: close-figs

        >>> pc = plot_rank_dist(centered, coords=coords, compact=False, combined=False)

    Similarly to the other ``combined=True`` case, the aesthetics stay the same
    as with ``combined=False``, but they are ignored by default when plotting
    on the left column.

    .. plot::
        :context: close-figs

        >>> pc = plot_rank_dist(centered, coords=coords, compact=False, combined=True)
        >>> pc.add_legend("chain")

    References
    ----------
    .. [1] Säilynoja et al. *Graphical test for discrete uniformity and
       its applications in goodness-of-fit evaluation and multiple sample comparison*.
       Statistics and Computing 32(32). (2022) https://doi.org/10.1007/s11222-022-10090-6

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if stats_dist_kwargs is None:
        stats_dist_kwargs = {}
    if stats_rank_kwargs is None:
        stats_rank_kwargs = {}
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
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", ["column"])
        aux_dim_list = [dim for dim in distribution.dims if dim not in sample_dims]
        if compact:
            pc_kwargs["rows"] = ["__variable__"]
        else:
            pc_kwargs.setdefault("rows", ["__variable__"] + aux_dim_list)
            aux_dim_list = [dim for dim in pc_kwargs["rows"] if dim != "__variable__"]
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

        pc_kwargs = set_grid_layout(pc_kwargs, plot_bknd, distribution, num_cols=2)

        plot_collection = PlotCollection.grid(
            distribution.expand_dims(column=2).assign_coords(column=["dist", "rank"]),
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
    aes_map["rank"] = {"overlay"}.union(aes_map.get("rank", plot_collection.aes_set))
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

    _, dist_aes, _ = filter_aes(plot_collection, aes_map, kind, sample_dims)

    # dens
    plot_kwargs_dist = {
        key: False
        for key in ("credible_interval", "point_estimate", "point_estimate_text", "title")
    }
    dist_kwargs = copy(plot_kwargs.get(kind, {}))
    if dist_kwargs is not False:
        if neutral_color and "color" not in dist_aes:
            dist_kwargs.setdefault("color", neutral_color)
        if neutral_linestyle and "linestyle" not in dist_aes:
            dist_kwargs.setdefault("linestyle", neutral_linestyle)
    plot_kwargs_dist[kind] = dist_kwargs
    if "remove_axis" in plot_kwargs:
        plot_kwargs_dist["remove_axis"] = plot_kwargs["remove_axis"]
    plot_collection.coords = {"column": "dist"}
    plot_dist(
        dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        kind=kind,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_map={key: value for key, value in aes_map.items() if key == kind},
        plot_kwargs=plot_kwargs_dist,
        stats_kwargs=stats_dist_kwargs,
    )
    plot_collection.coords = None

    # rank
    rank_kwargs = copy(plot_kwargs.get("rank", {}))
    div_kwargs = copy(plot_kwargs.get("divergence", {}))
    xlabel_kwargs = copy(plot_kwargs.get("xlabel_rank", {}))
    plot_kwargs_rank = {"rank": rank_kwargs, "divergence": div_kwargs, "xlabel": xlabel_kwargs}
    plot_kwargs_rank["title"] = False
    plot_kwargs_rank["ticklabels"] = False
    aes_map_rank = {
        key.replace("_rank", ""): value
        for key, value in plot_kwargs.items()
        if key in {"rank", "divergence", "xlabel_rank"}
    }
    plot_collection.coords = {"column": "rank"}
    plot_rank(
        dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_map=aes_map_rank,
        plot_kwargs=plot_kwargs_rank,
        stats_kwargs=stats_rank_kwargs,
    )
    plot_collection.coords = None
    if xlabel_kwargs is not False:
        plot_collection.rename_artists(xlabel="xlabel_rank")
    # divergences
    sample_stats = get_group(dt, "sample_stats", allow_missing=True)
    if (
        div_kwargs is not False
        and sample_stats is not None
        and "diverging" in sample_stats.data_vars
        and np.any(sample_stats.diverging)
    ):
        divergence_mask = dt.sample_stats.diverging
        _, div_aes, div_ignore = filter_aes(plot_collection, aes_map, "divergence", sample_dims)
        if "color" not in div_aes:
            div_kwargs.setdefault("color", "black")
        if "marker" not in div_aes:
            div_kwargs.setdefault("marker", "|")
        if "size" not in div_aes:
            div_kwargs.setdefault("size", 30)

        plot_collection.map(
            trace_rug,
            "divergence_dist",
            data=distribution,
            ignore_aes=div_ignore,
            xname=False,
            y=0,
            coords={"column": "dist"},
            mask=divergence_mask,
            **div_kwargs,
        )

    ## aesthetics
    # Add varnames as x and y labels
    _, labels_aes, labels_ignore = filter_aes(plot_collection, aes_map, "label", sample_dims)
    label_kwargs = copy(plot_kwargs.get("label", {}))
    if label_kwargs is not False:
        if "color" not in labels_aes:
            label_kwargs.setdefault("color", "black")

        plot_collection.map(
            labelled_x,
            "xlabel_dist",
            ignore_aes=labels_ignore,
            coords={"column": "dist"},
            subset_info=True,
            labeller=labeller,
            store_artist=backend == "none",
            **label_kwargs,
        )

        plot_collection.map(
            labelled_y,
            "ylabel_rank",
            ignore_aes=labels_ignore,
            coords={"column": "rank"},
            subset_info=True,
            labeller=labeller,
            store_artist=backend == "none",
            **label_kwargs,
        )

    # Adjust tick labels
    ticklabels_kwargs = copy(plot_kwargs.get("ticklabels", {}))
    if ticklabels_kwargs is not False:
        _, _, ticklabels_ignore = filter_aes(plot_collection, aes_map, "ticklabels", sample_dims)
        plot_collection.map(
            ticklabel_props,
            ignore_aes=ticklabels_ignore,
            axis="both",
            store_artist=backend == "none",
            **ticklabels_kwargs,
        )

    return plot_collection
