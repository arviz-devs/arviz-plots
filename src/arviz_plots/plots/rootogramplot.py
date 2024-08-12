"""rootogram plot code."""

from copy import copy
from importlib import import_module

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.base import array_stats

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import hist, labelled_title, line_xy, scatter_xy, trace_rug


def plot_rootogram(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    observed=None,
    observed_rug=False,
    coords=None,
    sample_dims=None,
    facet_dims=None,
    data_pairs=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot discrete prior/posterior predictive and observed values as rootogram plots.

    Parameters
    ----------
    dt : DataTree
        Input data with observed_data and prior/posterior predictive data.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default=None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted. Note: Posterior refers to posterior-predictive, prior refers to
        prior-predictive.
    observed : boolean, optional
        Whether or not to plot the observed data. Defaults to True for ``group = posterior``
        and False for ``group = prior``.
    observed_rug : boolean, default False
        Whether or not to plot a rug plot of the observed data. Only valid if observed=True.
    coords : dict, optional
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for that dimension.
        Defaults to including all coordinates for all dimensions if None.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Note: Dims not in sample_dims or facet_dims (below) will also be reduced by default.
        Defaults to ``rcParams["data.sample_dims"]``
    facet_dims : list, optional
        Dimensions to facet over (for which multiple plots will be generated).
        Defaults to empty list.
    data_pairs : dict, optional
        Dictionary containing relations between observed data and posterior/prior predictive data.
        Dictionary keys are variable names corresponding to observed data and dictionary values
        are variable names corresponding to posterior/prior predictive.
        For example, data_pairs = {'y' : 'y_hat'}.
        By default, it will assume that the observed data and the posterior/prior predictive data
        have the same variable names
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.
    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:
        * "predictive" -> Passed to :func:`~arviz_plots.visuals.hist_line`
        * "observed" -> passed to :func: `~arviz_plots.visuals.scatter_xy`
        * "observed_line" -> passed to :func:`~arviz_plots.visuals.line_xy`
        * "observed_rug" -> passed to :func:`arviz_plots.visuals.trace_rug`
        * "title" -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * "remove_axis" -> not passed anywhere, can only be ``False`` to skip calling this function
    stats_kwargs : mapping, optional
        Valid keys are:
        * predictive -> passed to hist
        * observed -> passed to hist
    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.wrap`
    Returns
    -------
    PlotCollection
    Examples
    --------
    WIP
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

    if stats_kwargs is None:
        stats_kwargs = {}
    if facet_dims is None:
        facet_dims = []

    # check for duplication of facet_dims in top level arg input and pc_kwargs
    if "cols" in pc_kwargs and len(facet_dims) > 0:
        raise ValueError(
            f"""Facet dimensions have been defined twice.
                Please pick only one of `facet_dims` or `pc_kwargs['cols']`.
                Currently defined facet_dims = {facet_dims}
                Currently defined pc_kwargs['cols'] = {pc_kwargs["cols"]}"""
        )

    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    predictive_data_group = f"{group}_predictive"
    if observed is None:
        observed = group == "posterior"  # by default true if posterior, false if prior

    # checking to make sure plot_kwargs["observed"] is not False
    observed_kwargs = copy(plot_kwargs.get("observed", {}))
    if observed_kwargs is False:
        raise ValueError(
            """plot_kwargs['observed'] can't be False, use observed=False to remove observed 
            plot element"""
        )

    # making sure both posterior/prior predictive group and observed_data group exists in
    # datatree provided
    if observed:
        for group_name in (predictive_data_group, "observed_data"):
            if group_name not in dt.children:
                raise TypeError(f'`data` argument must have the group "{group_name}" for ppcplot')
    else:
        if f"{predictive_data_group}" not in dt.children:
            raise TypeError(f'`data` argument must have the group "{group}_predictive" for ppcplot')

    # initializaing data_pairs as empty dict in case pp and observed data var names are same
    if data_pairs is None:
        data_pairs = {}

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    # getting plot backend to get default linestyles to put in default kwargs for visual elements
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    # pp distribution group plotting logic
    pp_distribution = process_group_variables_coords(
        dt,
        group=predictive_data_group,
        var_names=(
            None
            if var_names is None
            else [data_pairs.get(var_name, var_name) for var_name in var_names]
        ),
        filter_vars=filter_vars,
        coords=coords,
    )
    # print(f"\npp_distribution = {pp_distribution}")

    total_pp_samples = np.prod(
        [pp_distribution.sizes[dim] for dim in sample_dims if dim in pp_distribution.dims]
    )

    # wrap plot collection with pp distribution
    if plot_collection is None:
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]  # special variable to create one plot per variable
            + list(facet_dims),  # making sure multiple plots are created for each facet dim
        )
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("overlay", sample_dims)  # setting overlay dim
        plot_collection = PlotCollection.wrap(
            pp_distribution,
            backend=backend,
            **pc_kwargs,
        )
    # set reduce_dims by elimination
    reduce_dims = [
        dim for dim in pp_distribution.dims if dim not in set(facet_dims).union(set(sample_dims))
    ]

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("predictive", plot_collection.aes_set)
    if labeller is None:
        labeller = BaseLabeller()

    # setting plot_kwargs_dist defaults (for passing to internal plot_dist calls)
    plot_kwargs_dist = {
        key: False
        for key in ("credible_interval", "point_estimate", "point_estimate_text", "title")
    }
    # if "remove_axis" in plot_kwargs:
    plot_kwargs_dist["remove_axis"] = False  # plot_kwargs["remove_axis"]

    # print(f"\n aes_map = {aes_map}")

    # obs distribution calculated outside `if observed` since plotting predictive bars requires it
    observed_data_group = "observed_data"
    obs_distribution = process_group_variables_coords(
        dt,
        group=observed_data_group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
    )
    # print(f"\n obs_distribution = {obs_distribution}")

    # ---------(observed data)-----------
    # observed data calculations are made outside of and before 'if observed' since predictive also
    # depends on this computed data (number of bins and top of predictive bars for rootograms)

    # use get_bins func from arviz-stats on observed data and then use those bins for
    # computing histograms for predictive data as well
    # WIP: currently only the bins for one variable (without any facetting) is retrieved and used
    bins = array_stats.get_bins(obs_distribution["home_points"].values)
    print(f"\n bins = {bins}")

    # this portion is situated in an out of convention spot becuse obs_hist_dims is required
    obs_hist_dims, obs_hist_aes, obs_hist_ignore = filter_aes(
        plot_collection, aes_map, "observed", reduce_dims
    )

    obs_stats_kwargs = copy(stats_kwargs.get("observed", {}))
    obs_stats_kwargs.setdefault("bins", bins)

    obs_hist = obs_distribution.azstats.histogram(
        dims=list(obs_hist_dims) + list(sample_dims), **obs_stats_kwargs
    )

    # print(f"\n obs_hist = {obs_hist}")

    obs_hist.loc[{"plot_axis": "histogram"}] = (obs_hist.sel(plot_axis="histogram")) ** 0.5

    # print(f"\n obs_density.data_vars = {obs_density.data_vars}")
    # print(f"\n obs_density.keys() = {obs_density.keys()}")

    # new_obs_hist with histogram->y and left_edge/right_edge midpoint->x
    new_obs_hist = xr.Dataset()

    for var_name in list(obs_hist.keys()):
        left_edges = obs_hist[var_name].sel(plot_axis="left_edges").values
        right_edges = obs_hist[var_name].sel(plot_axis="right_edges").values

        left_edges = np.array(left_edges)
        right_edges = np.array(right_edges)

        # print(f"\n left_edges = {left_edges}")
        # print(f"\n right_edges = {right_edges}")

        x = (left_edges + right_edges) / 2
        y = obs_hist[var_name].sel(plot_axis="histogram").values

        # print(f"\n new_obs_hist y= {y}")
        # print(f"x = {x} | y = {y}")

        stacked_data = np.stack((x, y), axis=-1)
        new_var = xr.DataArray(
            stacked_data, dims=["hist_dim", "plot_axis"], coords={"plot_axis": ["x", "y"]}
        )

        new_obs_hist[var_name] = new_var

    print(f"\n new_obs_hist = {new_obs_hist}")

    if observed:  # all observed data group plotting logic happens here
        obs_kwargs = copy(plot_kwargs.get("observed", {}))
        if obs_kwargs is not False:
            if "color" not in obs_hist_aes:
                obs_kwargs.setdefault("color", "black")

            plot_collection.map(
                scatter_xy,
                "observed",
                data=new_obs_hist,
                ignore_aes=obs_hist_ignore,
                **obs_kwargs,
            )

        obs_line_kwargs = copy(plot_kwargs.get("observed_line", {}))
        if obs_line_kwargs is not False:
            _, obs_line_aes, obs_line_ignore = filter_aes(
                plot_collection, aes_map, "observed", reduce_dims
            )

            linestyle = plot_bknd.get_default_aes("linestyle", 2, {})[1]

            if "linestyle" not in obs_line_aes:
                obs_kwargs.setdefault("linestyle", linestyle)

            if "color" not in obs_line_aes:
                obs_line_kwargs.setdefault("color", "black")

            plot_collection.map(
                line_xy,
                "observed_line",
                data=new_obs_hist,
                ignore_aes=obs_line_ignore,
                **obs_line_kwargs,
            )

        # ---------(observed rug plot)-----------
        if observed_rug:
            # plot observed density as a rug
            rug_kwargs = copy(plot_kwargs.get("observed_rug", {}))

            _, rug_aes, rug_ignore = filter_aes(
                plot_collection, aes_map, "observed_rug", reduce_dims
            )
            if "color" not in rug_aes:
                rug_kwargs.setdefault("color", "black")
            if "marker" not in rug_aes:
                rug_kwargs.setdefault("marker", "|")
            if "size" not in rug_aes:
                rug_kwargs.setdefault("size", 30)

            # print(f"\nobs_distribution = {obs_distribution}")

            plot_collection.map(
                trace_rug,
                "observed_rug",
                data=obs_distribution,
                ignore_aes=rug_ignore,
                xname=False,
                y=0,
                **rug_kwargs,
            )

    # ---------(PPC data)-------------

    pp_kwargs = copy(plot_kwargs.get("predictive", {}))

    if pp_kwargs is not False:
        pp_hist_dims, pp_hist_aes, pp_hist_ignore = filter_aes(
            plot_collection, aes_map, "predictive", reduce_dims
        )
        # getting first default color from color cycle and picking it
        pp_default_color = plot_bknd.get_default_aes("color", 1, {})[0]
        if "color" not in pp_hist_aes:
            pp_kwargs.setdefault("color", pp_default_color)

        if "alpha" not in pp_hist_aes:
            pp_kwargs.setdefault("alpha", 0.2)

        pp_stats_kwargs = copy(stats_kwargs.get("predictive", {}))
        pp_stats_kwargs.setdefault("bins", bins)

        pp_hist = pp_distribution.azstats.histogram(
            dims=list(pp_hist_dims) + list(sample_dims), **pp_stats_kwargs
        )

        # print(f"\n pp_density histogram form pp_hist = {pp_hist}")

        # the top of the predictive bars height = the observed height for that bin
        # the bottom = difference between observed and predictive height for that bin

        # getting mean of counts across all predictive samples and taking its square root
        pp_hist.loc[{"plot_axis": "histogram"}] = (
            pp_hist.sel(plot_axis="histogram") / total_pp_samples
        ) ** 0.5

        # new_pp_hist dataset
        new_pp_hist = xr.Dataset()

        for var_name in list(pp_hist.keys()):
            left_edges = pp_hist[var_name].sel(plot_axis="left_edges").values
            right_edges = pp_hist[var_name].sel(plot_axis="right_edges").values

            # getting top of histogram (observed values dataset's 'y' coord)
            new_histogram = new_obs_hist[var_name].sel(plot_axis="y").values
            print(f"\n new_pp_hist (new_histogram) observed heights= {new_histogram}")

            histogram_bottom = new_histogram - pp_hist[var_name].sel(plot_axis="histogram").values
            # print(f"\n new_pp_hist histogram_bottom= {histogram_bottom}")

            stacked_data = np.stack(
                (new_histogram, left_edges, right_edges, histogram_bottom), axis=-1
            )
            new_var = xr.DataArray(
                stacked_data,
                dims=["hist_dim", "plot_axis"],
                coords={
                    "plot_axis": ["histogram", "left_edges", "right_edges", "histogram_bottom"]
                },
            )

            new_pp_hist[var_name] = new_var

        print(f"\n new_pp_hist = {new_pp_hist}")

        plot_collection.map(
            hist, "predictive", data=new_pp_hist, ignore_aes=pp_hist_ignore, **pp_kwargs
        )

    # adding baseline at 0
    baseline_kwargs = copy(plot_kwargs.get("baseline", {}))
    if baseline_kwargs is not False:
        _, baseline_aes, baseline_ignore = filter_aes(
            plot_collection, aes_map, "baseline", reduce_dims
        )

        linestyle = plot_bknd.get_default_aes("linestyle", 2, {})[1]

        if "linestyle" not in baseline_aes:
            baseline_kwargs.setdefault("linestyle", linestyle)

        if "color" not in baseline_aes:
            baseline_kwargs.setdefault("color", "black")

        plot_collection.map(
            line_xy,
            "baseline",
            data=obs_distribution,
            x=bins,
            y=0,
            ignore_aes=baseline_ignore,
            **baseline_kwargs,
        )

    # adding plot title/s
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

    # print(f"\nsample_dims = {sample_dims}")
    # print(f"\nfacet_dims = {facet_dims}")
    # print(f"\nreduce_dims = {reduce_dims}")

    # print(f"\n-----------------------------------------------------------------\n")
    # print(f"\n datatree = {dt}")
    # print(f"\n pc.viz = {plot_collection.viz!r}")
    # print(f"\n pc.aes = {plot_collection.aes!r}")
    # print(f"\n-----------------------------------------------------------------\n")

    return plot_collection
