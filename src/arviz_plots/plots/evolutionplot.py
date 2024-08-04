"""evolution ess plot."""

# imports
# import warnings
from copy import copy
from importlib import import_module

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, get_group, process_group_variables_coords
from arviz_plots.visuals import (
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
    trace_rug,
)


def plot_ess_evolution(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    relative=False,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    extra_methods=False,
    min_ess=400,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot estimated effective sample size plots for increasing number of iterations.

    Parameters
    ----------
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.
    var_names : str or sequence of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default None
        If None, interpret `var_names` as the real variables names.
        If “like”, interpret `var_names` as substrings of the real variables names.
        If “regex”, interpret `var_names` as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    relative : bool, default False
        Show relative ess in plot ``ress = ess / N``.
    rug : bool, default False
        Add a `rug plot <https://en.wikipedia.org/wiki/Rug_plot>`_ for a specific subset of values.
    rug_kind : str, default "diverging"
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points : int, default 20
        Number of subsets in the evolution plot.
    extra_methods : bool, default False
        Plot mean and sd ESS as horizontal lines.
    min_ess : int, default 400
        Minimum number of ESS desired. If ``relative=True`` the line is plotted at
        ``min_ess / n_samples`` as a curve following the ``min_ess / n`` dependency
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str or False}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:
        * ess_bulk -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * ess_bulk_line -> passed to :func:`~arviz_plots.visuals.line_xy`
        * ess_tail -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * ess_tail_line -> passed to :func:`~arviz_plots.visuals.line_xy`
        * rug -> passed to :func:`~.visuals.trace_rug`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * mean -> passed to :func:`~arviz_plots.visuals.line_xy`
        * sd -> passed to :func:`~arviz_plots.visuals.line_xy`
        * min_ess -> passed to :func:`~arviz_plots.visuals.line_xy`

    stats_kwargs : mapping, optional
        Valid keys are:
        * ess_bulk -> passed to ess, method = 'bulk'
        * ess_tail -> passed to ess, method = 'tail'
        * mean -> passed to ess, method='mean'
        * sd -> passed to ess, method='sd'

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection
    """
    # initial defaults
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    # mutable inputs
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if stats_kwargs is None:
        stats_kwargs = {}

    # processing dt/group/coords/filtering
    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    # set plot collection initialization defaults if it doesnt exist
    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]
            + [dim for dim in distribution.dims if dim not in {"model"}.union(sample_dims)],
        )
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        aux_dim_list = [dim for dim in pc_kwargs["cols"] if dim != "__variable__"]  # for divergence
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

    # set plot collection dependent defaults (like aesthetics mappings for each artist)
    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("ess_bulk", plot_collection.aes_set.difference({"overlay"}))
    aes_map.setdefault("ess_bulk_line", plot_collection.aes_set.difference({"overlay"}))
    aes_map.setdefault("ess_tail", plot_collection.aes_set.difference({"overlay"}))
    aes_map.setdefault("ess_tail_line", plot_collection.aes_set.difference({"overlay"}))
    aes_map.setdefault("divergence", {"overlay"})
    if labeller is None:
        labeller = BaseLabeller()

    # compute and add ess evolution subplots for 'bulk' and 'tail'
    if len(sample_dims) > 1:
        n_samples = 1
        for dim in sample_dims:
            if dim in distribution:
                n_samples = distribution.sizes[dim] * n_samples
            n_draws = distribution.sizes[sample_dims[1]]  # second sample_dim as default draw dim
    else:
        n_samples = distribution.sizes[sample_dims[0]]
        n_draws = n_samples  # assuming only sample_dim to be draw dim

    # setting xdata and draw_divisions for later ess computing and plotting
    xdata = np.linspace(n_samples / n_points, n_samples, n_points)
    draw_divisions = np.linspace(n_draws // n_points, n_draws, n_points, dtype=int)

    # n_draws = distribution.sizes["draw"]
    # n_samples = n_draws * distribution.sizes["chain"]

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    default_bulk_color, default_tail_color = plot_bknd.get_default_aes("color", 2, {})

    ess_bulk_dataset = None

    # defining common ess_dataset computing function
    def compute_ess_dataset(
        distribution,
        xdata,
        draw_divisions,
        method,  # "bulk" or "tail"
        method_dims,  # bulk_dims or tail_dims
        relative,
        stats_kwargs,
    ):
        first_draw = distribution.draw.values[0]
        ess_y_dataset = xr.concat(
            [
                distribution.sel(draw=slice(first_draw + draw_div)).azstats.ess(
                    dims=method_dims,
                    method=method,
                    relative=relative,
                    **stats_kwargs.get(f"ess_{method}", {}),
                )
                for draw_div in draw_divisions
            ],
            dim="ess_dim",
        )

        # converting xdata into a xr dataarray
        xdata_da = xr.DataArray(xdata, dims="ess_dim")
        # broadcasting xdata_da to match shape of each variable in ess_y_dataset and
        # creating a new dataset from dict of broadcasted xdata
        xdata_dataset = xr.Dataset(
            {var_name: xdata_da.broadcast_like(da) for var_name, da in ess_y_dataset.items()}
        )
        # concatenating xdata_dataset and ess_y_dataset along plot_axis
        ess_dataset = xr.concat([xdata_dataset, ess_y_dataset], dim="plot_axis").assign_coords(
            plot_axis=["x", "y"]
        )

        return ess_dataset

    bulk_kwargs = copy(plot_kwargs.get("ess_bulk", {}))
    if bulk_kwargs is not False:
        bulk_dims, bulk_aes, bulk_ignore = filter_aes(
            plot_collection, aes_map, "ess_bulk", sample_dims
        )

        ess_bulk_dataset = compute_ess_dataset(
            distribution,
            xdata,
            draw_divisions,
            "bulk",
            bulk_dims,
            relative,
            stats_kwargs,
        )

        if "color" not in bulk_aes:
            bulk_kwargs.setdefault("color", default_bulk_color)

        plot_collection.map(
            scatter_xy, "ess_bulk", data=ess_bulk_dataset, ignore_aes=bulk_ignore, **bulk_kwargs
        )

    bulk_line_kwargs = copy(plot_kwargs.get("ess_bulk_line", {}))
    if bulk_line_kwargs is not False:
        bulk_line_dims, bulk_line_aes, bulk_line_ignore = filter_aes(
            plot_collection, aes_map, "ess_bulk_line", sample_dims
        )

        if ess_bulk_dataset is None:
            ess_bulk_dataset = compute_ess_dataset(
                distribution,
                xdata,
                draw_divisions,
                "bulk",
                bulk_line_dims,
                relative,
                stats_kwargs,
            )

        if "color" not in bulk_line_aes:
            bulk_line_kwargs.setdefault("color", default_bulk_color)

        plot_collection.map(
            line_xy,
            "ess_bulk_line",
            data=ess_bulk_dataset,
            ignore_aes=bulk_line_ignore,
            **bulk_line_kwargs,
        )

    ess_tail_dataset = None

    tail_kwargs = copy(plot_kwargs.get("ess_tail", {}))

    if tail_kwargs is not False:
        tail_dims, tail_aes, tail_ignore = filter_aes(
            plot_collection, aes_map, "ess_tail", sample_dims
        )

        ess_tail_dataset = compute_ess_dataset(
            distribution,
            xdata,
            draw_divisions,
            "tail",
            tail_dims,
            relative,
            stats_kwargs,
        )

        if "color" not in tail_aes:
            tail_kwargs.setdefault("color", default_tail_color)

        plot_collection.map(
            scatter_xy, "ess_tail", data=ess_tail_dataset, ignore_aes=tail_ignore, **tail_kwargs
        )

    tail_line_kwargs = copy(plot_kwargs.get("ess_tail_line", {}))

    if tail_line_kwargs is not False:
        tail_line_dims, tail_line_aes, tail_line_ignore = filter_aes(
            plot_collection, aes_map, "ess_tail_line", sample_dims
        )

        if ess_tail_dataset is None:
            ess_tail_dataset = compute_ess_dataset(
                distribution,
                xdata,
                draw_divisions,
                "tail",
                tail_line_dims,
                relative,
                stats_kwargs,
            )

        if "color" not in tail_line_aes:
            tail_line_kwargs.setdefault("color", default_tail_color)

        plot_collection.map(
            line_xy,
            "ess_tail_line",
            data=ess_tail_dataset,
            ignore_aes=tail_line_ignore,
            **tail_line_kwargs,
        )

    # plot rug WIP: to use with rank_data from arviz-stats
    # overlaying divergences for each chain
    if rug:
        sample_stats = get_group(dt, "sample_stats", allow_missing=True)
        divergence_kwargs = copy(plot_kwargs.get("divergence", {}))
        if (
            sample_stats is not None
            and "diverging" in sample_stats.data_vars
            and np.any(sample_stats[rug_kind])  # 'diverging' by default
            and divergence_kwargs is not False
        ):
            divergence_mask = dt.sample_stats[rug_kind]  # 'diverging' by default
            print(f"\n divergence_mask = {divergence_mask!r}")
            _, div_aes, div_ignore = filter_aes(plot_collection, aes_map, "divergence", sample_dims)
            if "color" not in div_aes:
                divergence_kwargs.setdefault("color", "black")
            if "marker" not in div_aes:
                divergence_kwargs.setdefault("marker", "|")
            # if "width" not in div_aes: # should this be hardcoded?
            #    divergence_kwargs.setdefault("width", linewidth)
            if "size" not in div_aes:
                divergence_kwargs.setdefault("size", 30)
            div_reduce_dims = [dim for dim in distribution.dims if dim not in aux_dim_list]

            # xname is used to pick subset of dataset in map() to be masked
            xname = None  # xname logic from traceplot
            default_xname = sample_dims[0] if len(sample_dims) == 1 else "draw"
            if (default_xname not in distribution.dims) or (
                not np.issubdtype(distribution[default_xname].dtype, np.number)
            ):
                default_xname = None
            xname = divergence_kwargs.get("xname", default_xname)
            divergence_kwargs["xname"] = xname
            print(f"\n div_reduce_dims = {div_reduce_dims!r}")
            print(f"\n xname = {xname}")

            draw_length = distribution.sizes["draw"]  # used to scale xvalues to between 0-1

            # print(f"\n distribution = {distribution}")

            plot_collection.map(
                trace_rug,
                "divergence",
                data=distribution,
                ignore_aes=div_ignore,
                # xname=xname,
                y=distribution.min(div_reduce_dims),
                mask=divergence_mask,
                scale=draw_length,
                **divergence_kwargs,
            )

    # getting backend specific linestyles
    linestyles = plot_bknd.get_default_aes("linestyle", 4, {})
    # and default color
    default_color = plot_bknd.get_default_aes("color", 1, {})[0]

    # plot mean and sd
    if extra_methods is not False:
        mean_kwargs = copy(plot_kwargs.get("mean", {}))
        if mean_kwargs is not False:
            mean_dims, mean_aes, mean_ignore = filter_aes(
                plot_collection, aes_map, "mean", sample_dims
            )
            mean_ess = distribution.azstats.ess(
                dims=mean_dims, method="mean", relative=relative, **stats_kwargs.get("mean", {})
            )

            # getting 2nd default linestyle for chosen backend and assigning it by default
            mean_kwargs.setdefault("linestyle", linestyles[1])

            if "color" not in mean_aes:
                mean_kwargs.setdefault("color", default_color)

            plot_collection.map(
                line_xy,
                "mean",
                data=mean_ess,
                x=xdata,
                ignore_aes=mean_ignore,
                **mean_kwargs,
            )

        sd_kwargs = copy(plot_kwargs.get("sd", {}))
        if sd_kwargs is not False:
            sd_dims, sd_aes, sd_ignore = filter_aes(plot_collection, aes_map, "sd", sample_dims)
            sd_ess = distribution.azstats.ess(
                dims=sd_dims, method="sd", relative=relative, **stats_kwargs.get("sd", {})
            )

            sd_kwargs.setdefault("linestyle", linestyles[2])

            if "color" not in sd_aes:
                sd_kwargs.setdefault("color", default_color)

            plot_collection.map(
                line_xy, "sd", data=sd_ess, ignore_aes=sd_ignore, x=xdata, **sd_kwargs
            )

    # plot minimum ess
    min_ess_kwargs = copy(plot_kwargs.get("min_ess", {}))

    if min_ess_kwargs is not False:
        _, min_ess_aes, min_ess_ignore = filter_aes(
            plot_collection, aes_map, "min_ess", sample_dims
        )

        if relative:
            min_ess = min_ess / n_points

        min_ess_kwargs.setdefault("linestyle", linestyles[3])

        if "color" not in min_ess_aes:
            min_ess_kwargs.setdefault("color", "gray")

        plot_collection.map(
            line_xy,
            "min_ess",
            data=distribution,
            ignore_aes=min_ess_ignore,
            x=xdata,
            y=min_ess,
            **min_ess_kwargs,
        )

    # plot titles for each facetted subplot
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

    # plot x and y axis labels
    # Add varnames as x and y labels
    _, xlabels_aes, xlabels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = plot_kwargs.get("xlabel", {}).copy()
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "black")

        xlabel_kwargs.setdefault(
            "text", sample_dims[0] if len(sample_dims) == 1 else "Total Number of Draws"
        )

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=xlabels_ignore,
            subset_info=True,
            store_artist=False,
            **xlabel_kwargs,
        )

    _, ylabels_aes, ylabels_ignore = filter_aes(plot_collection, aes_map, "ylabel", sample_dims)
    ylabel_kwargs = plot_kwargs.get("ylabel", {}).copy()
    if ylabel_kwargs is not False:
        if "color" not in ylabels_aes:
            ylabel_kwargs.setdefault("color", "black")

        ylabel = "{}"
        ylabel_kwargs.setdefault(
            "text",
            ylabel.format("Relative ESS") if relative is not False else ylabel.format("ESS"),
        )

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=ylabels_ignore,
            subset_info=True,
            store_artist=False,
            **ylabel_kwargs,
        )

    return plot_collection
