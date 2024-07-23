"""ess plot code."""

# imports
# import warnings
from copy import copy

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


def plot_ess(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind="local",
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
    """Plot effective sample size plots.

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
    kind : {"local", "quantile", "evolution"}, default "local"
        Specify the kind of plot:

        * The ``kind="local"`` argument generates the ESS' local efficiency
          for estimating small-interval probability of a desired posterior.
        * The ``kind="quantile"`` argument generates the ESS' local efficiency
          for estimating quantiles of a desired posterior.

    relative : bool, default False
        Show relative ess in plot ``ress = ess / N``.
    rug : bool, default False
        Add a `rug plot <https://en.wikipedia.org/wiki/Rug_plot>`_ for a specific subset of values.
    rug_kind : str, default "diverging"
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points : int, default 20
        Number of points for which to plot their quantile/local ess or number of subsets
        in the evolution plot.
    extra_methods : bool, default False
        Plot mean and sd ESS as horizontal lines.
    min_ess : int, default 400
        Minimum number of ESS desired. If ``relative=True`` the line is plotted at
        ``min_ess / n_samples`` for local and quantile kinds
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str or False}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * ess -> passed to :func:`~arviz_plots.visuals.scatter_xy`
            if `kind`='local',
            else passed to :func:`~arviz_plots.visuals.scatter_xy`
            if `kind` = 'quantile'

        * rug -> passed to :func:`~.visuals.trace_rug`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * mean -> passed to :func:`~arviz.plots.visuals.line_xy`
        * sd -> passed to :func:`~arviz.plots.visuals.line_xy`
        * min_ess -> passed to :func:`~arviz.plots.visuals.line_xy`

    stats_kwargs : mapping, optional
        Valid keys are:

        * ess -> passed to ess, method = 'local' or 'quantile' based on `kind`
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
        if "model" in distribution:
            pc_kwargs["aes"].setdefault("color", ["model"])
            n_models = distribution.sizes["model"]
            x_diff = min(1 / n_points / 3, 1 / n_points * n_models / 10)
            pc_kwargs.setdefault("x", np.linspace(-x_diff, x_diff, n_models))
            pc_kwargs["aes"].setdefault("x", ["model"])
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

    # set plot collection dependent defaults (like aesthetics mappings for each artist)
    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault(kind, plot_collection.aes_set.difference({"overlay"}))
    aes_map.setdefault("rug", {"overlay"})
    if labeller is None:
        labeller = BaseLabeller()

    # compute and add ess subplots
    ess_kwargs = copy(plot_kwargs.get("ess", {}))

    if ess_kwargs is not False:
        ess_dims, _, ess_ignore = filter_aes(plot_collection, aes_map, kind, sample_dims)
        if kind == "local":
            probs = np.linspace(0, 1, n_points, endpoint=False)
            ylabel = "{} for small intervals"
        elif kind == "quantile":
            probs = np.linspace(1 / n_points, 1 - 1 / n_points, n_points)
            ylabel = "{} for quantiles"
        xdata = probs

        ess_y_dataset = xr.concat(
            [
                distribution.azstats.ess(
                    dims=ess_dims,
                    method=kind,
                    relative=relative,
                    prob=[p, (p + 1 / n_points)] if kind == "local" else p,
                    **stats_kwargs.get("ess", {}),
                )
                for p in probs
            ],
            dim="ess_dim",
        )

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

        plot_collection.map(
            scatter_xy, "ess", data=ess_dataset, ignore_aes=ess_ignore, **ess_kwargs
        )

    # plot rug
    # overlaying divergences(or other 'rug_kind') for each chain
    if rug:
        sample_stats = get_group(dt, "sample_stats", allow_missing=True)
        rug_kwargs = copy(plot_kwargs.get("rug", {}))
        if rug_kwargs is False:
            raise ValueError("plot_kwargs['rug'] can't be False, use rug=False to remove the rug")
        if (
            sample_stats is not None
            and rug_kind in sample_stats.data_vars
            and np.any(sample_stats[rug_kind])  # 'diverging' by default
            and rug_kwargs is not False
        ):
            rug_mask = dt.sample_stats[rug_kind]  # 'diverging' by default
            _, div_aes, div_ignore = filter_aes(plot_collection, aes_map, "rug", sample_dims)
            if "color" not in div_aes:
                rug_kwargs.setdefault("color", "black")
            if "marker" not in div_aes:
                rug_kwargs.setdefault("marker", "|")
            # WIP: if using a default linewidth once defined in backend/agnostic defaults
            # if "width" not in div_aes:
            #    # get default linewidth for backends
            #    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
            #    default_linewidth = plot_bknd.get_default_aes("linewidth", 1, {})
            #    rug_kwargs.setdefault("width", default_linewidth)
            if "size" not in div_aes:
                rug_kwargs.setdefault("size", 30)
            div_reduce_dims = [dim for dim in distribution.dims if dim not in aux_dim_list]

            # xname is used to pick subset of dataset in map() to be masked
            xname = None
            default_xname = sample_dims[0] if len(sample_dims) == 1 else "draw"
            if (default_xname not in distribution.dims) or (
                not np.issubdtype(distribution[default_xname].dtype, np.number)
            ):
                default_xname = None
            xname = rug_kwargs.get("xname", default_xname)
            rug_kwargs["xname"] = xname

            draw_length = (
                distribution.sizes[sample_dims[0]]
                if len(sample_dims) == 1
                else distribution.sizes["draw"]
            )  # used to scale xvalues to between 0-1

            plot_collection.map(
                trace_rug,
                "rug",
                data=distribution,
                ignore_aes=div_ignore,
                # xname=xname,
                y=distribution.min(div_reduce_dims),
                mask=rug_mask,
                scale=draw_length,
                **rug_kwargs,
            )  # note: after plot_ppc merge, the `trace_rug` function might change

    # defining x_range (used for mean, sd, minimum ess plotting)
    x_range = [0, 1]
    x_range = xr.DataArray(x_range)

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
            print(f"\n mean_ess = {mean_ess}")

            if "linestyle" not in mean_aes:
                if backend == "matplotlib":
                    mean_kwargs.setdefault("linestyle", "--")
                elif backend == "bokeh":
                    mean_kwargs.setdefault("linestyle", "dashed")

            plot_collection.map(
                line_xy,
                "mean",
                data=mean_ess,
                x=x_range,
                ignore_aes=mean_ignore,
                **mean_kwargs,
            )

        sd_kwargs = copy(plot_kwargs.get("sd", {}))
        if sd_kwargs is not False:
            sd_dims, sd_aes, sd_ignore = filter_aes(plot_collection, aes_map, "sd", sample_dims)
            sd_ess = distribution.azstats.ess(
                dims=sd_dims, method="sd", relative=relative, **stats_kwargs.get("sd", {})
            )
            print(f"\n sd_ess = {sd_ess}")

            if "linestyle" not in sd_aes:
                if backend == "matplotlib":
                    sd_kwargs.setdefault("linestyle", "--")
                elif backend == "bokeh":
                    sd_kwargs.setdefault("linestyle", "dashed")

            plot_collection.map(
                line_xy, "sd", data=sd_ess, ignore_aes=sd_ignore, x=x_range, **sd_kwargs
            )

    # plot minimum ess
    min_ess_kwargs = copy(plot_kwargs.get("min_ess", {}))

    if min_ess_kwargs is not False:
        min_ess_dims, min_ess_aes, min_ess_ignore = filter_aes(
            plot_collection, aes_map, "min_ess", sample_dims
        )

        if relative:
            min_ess = min_ess / n_points

        # for each variable of distribution, put min_ess as the value, reducing all min_ess_dims
        min_ess_data = {}
        for var in distribution.data_vars:
            reduced_data = distribution[var].mean(
                dim=[dim for dim in distribution[var].dims if dim in min_ess_dims]
            )
            min_ess_data[var] = xr.full_like(reduced_data, min_ess)

        min_ess_dataset = xr.Dataset(min_ess_data)
        print(f"\n min_ess = {min_ess_dataset}")

        if "linestyle" not in min_ess_aes:
            if backend == "matplotlib":
                min_ess_kwargs.setdefault("linestyle", "--")
            elif backend == "bokeh":
                min_ess_kwargs.setdefault("linestyle", "dashed")

        plot_collection.map(
            line_xy,
            "min_ess",
            data=min_ess_dataset,
            ignore_aes=min_ess_ignore,
            x=x_range,
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
    _, labels_aes, labels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = plot_kwargs.get("xlabel", {}).copy()
    if xlabel_kwargs is not False:
        if "color" not in labels_aes:
            xlabel_kwargs.setdefault("color", "black")

        # formatting ylabel and setting xlabel
        xlabel_kwargs.setdefault("text", "Quantile")

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=labels_ignore,
            subset_info=True,
            store_artist=False,
            **xlabel_kwargs,
        )

    _, labels_aes, labels_ignore = filter_aes(plot_collection, aes_map, "ylabel", sample_dims)
    ylabel_kwargs = plot_kwargs.get("ylabel", {}).copy()
    if ylabel_kwargs is not False:
        if "color" not in labels_aes:
            ylabel_kwargs.setdefault("color", "black")

        if relative is not False:
            ylabel_text = ylabel.format("Relative ESS")
        else:
            ylabel_text = ylabel.format("ESS")
        ylabel_kwargs.setdefault("text", ylabel_text)

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=labels_ignore,
            subset_info=True,
            store_artist=False,
            **ylabel_kwargs,
        )

    return plot_collection
