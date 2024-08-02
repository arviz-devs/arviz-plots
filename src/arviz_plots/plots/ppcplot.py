"""ppc plot code."""

from copy import copy
from importlib import import_module
from numbers import Integral

import arviz_stats  # pylint: disable=unused-import
import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from datatree import DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.distplot import plot_dist
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import labelled_title, trace_rug


# define function
def plot_ppc(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    observed=None,
    observed_rug=False,
    coords=None,
    sample_dims=None,
    kind=None,
    facet_dims=None,
    data_pairs=None,
    aggregate=False,
    num_pp_samples=None,
    random_seed=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot prior/posterior predictive and observed values as kde, ecdf, hist, or scatter plots.

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
        Dimensions to loop over in plotting posterior/prior predictive.
        Note: Dims not in sample_dims or facet_dims (below) will be reduced by default.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "ecdf", "hist, "scatter"}, optional
        How to represent the marginal density. Defaults to ``rcParams["plot.density_kind"]``.
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
    aggregate: bool, default False
        If True, predictive data will be aggregated over both sample_dims and reduce_dims.
    num_pp_samples : int, optional
        Number of prior/posterior predictive samples to plot.
        Defaults to the total sample size (product of sample_dim dimension lengths) or minimum
        between total sample size and 5 in case of kind='scatter'
    random_seed : int, optional
        Random number generator seed passed to numpy.random.seed to allow reproducibility of the
        plot.
        By default, no seed will be provided and the plot will change each call if a random sample
        is specified by num_pp_samples.
    jitter : float, optional
        If kind is “scatter”, jitter will add random uniform noise to the height of the ppc samples
        and observed data.
    animated : bool, optional
        Create an animation of one posterior/prior predictive sample per frame if true.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * "predictive" -> Passed to either of "kde", "ecdf", "hist", "scatter" based on `kind`
        * "observed" -> passed to either of "kde", "ecdf", "hist", "scatter" based on `kind`
        * "aggregate" -> passed to either of "kde", "ecdf", "hist", "scatter" based on `kind`

        Values of the above plot_kwargs keys are passed to one of "kde", "ecdf", "hist", "scatter",
        matching the `kind` argument.

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func:`~arviz_plots.visuals.hist_line`
          * "scatter" -> passed to :func: `~arviz_plots.visuals.scatter_x`

        * "observed_rug" -> passed to :func:`arviz_plots.visuals.trace_rug`
        * "title" -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * "remove_axis" -> not passed anywhere, can only be ``False`` to skip calling this function

    stats_kwargs : mapping, optional
        Valid keys are:

        * predictive -> passed to kde, ecdf, ...
        * aggregate -> passed to kde, ecdf, ...
        * observed -> passed to kde, ecdf, ...

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
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if plot_kwargs is None:
        plot_kwargs = {}
    # Note: This function internally calls `plot_dist` so the 3 relevant artists "predictive",
    # "aggregate", "observed" get mapped to one of `plot_dist`'s density artists- "kde",
    # "ecdf", "scatter" based on the value of the top level arg `kind`
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

    # checking to make sure plot_kwargs["observed"] or plot_kwargs["aggregate"] are not False
    observed_kwargs = copy(plot_kwargs.get("observed", {}))
    if observed_kwargs is False:
        raise ValueError(
            """plot_kwargs['observed'] can't be False, use observed=False to remove observed 
            plot element"""
        )

    aggregate_kwargs = copy(plot_kwargs.get("aggregate", {}))
    if aggregate_kwargs is False:
        raise ValueError(
            """plot_kwargs['aggregate'] can't be False, use aggregate=False to remove 
            aggregate plot element"""
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

    # making sure kde type is one of these three
    if kind.lower() not in ("kde", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `cumulative`, or `scatter`")
    if kind == "cumulative":
        kind = "ecdf"

    # initializaing data_pairs as empty dict in case pp and observed data var names are same
    if data_pairs is None:
        data_pairs = {}

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend
    # getting plot backend to get default colors to put in default kwargs for visual elements
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

    # creating random number generator
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # picking random pp dataset sample indexes and subsetting pp_distribution accordingly
    total_pp_samples = np.prod(
        [pp_distribution.sizes[dim] for dim in sample_dims if dim in pp_distribution.dims]
    )
    if num_pp_samples is None:
        if kind == "scatter":
            num_pp_samples = min(5, total_pp_samples)
        else:
            num_pp_samples = total_pp_samples

    if (
        not isinstance(num_pp_samples, Integral)
        or num_pp_samples < 1
        or num_pp_samples > total_pp_samples
    ):
        raise TypeError(f"`num_pp_samples` must be an integer between 1 and {total_pp_samples}.")

    # stacking sample dimensions and selecting randomly if sample_dims length>1 or
    # num_pp_samples=total_pp_samples
    if num_pp_samples != total_pp_samples or len(sample_dims) == 1:
        pp_sample_ix = rng.choice(total_pp_samples, size=num_pp_samples, replace=False)

        # stacking sample dims into a new 'ppc_dim' dimension
        pp_distribution = pp_distribution.stack(ppc_dim=sample_dims)

        # Select the desired samples
        pp_distribution = pp_distribution.isel(ppc_dim=pp_sample_ix)

        # renaming sample_dims so that rest of plot will consider this as sample_dims
        sample_dims = ["ppc_dim"]

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
    # setting aggregate aes_map to `[]` so `overlay` isn't applied for it
    aes_map.setdefault("aggregate", [])
    if labeller is None:
        labeller = BaseLabeller()

    # checking plot_collection wrapped dataset and viz/aes datatrees
    # print(f"\nplot_collection.data = {plot_collection.data}")
    # print(f"\nplot_collection.aes = {plot_collection.aes}")
    # print(f"\nplot_collection.viz = {plot_collection.viz}")

    # setting plot_kwargs_dist defaults (for passing to internal plot_dist calls)
    plot_kwargs_dist = {
        key: False
        for key in ("credible_interval", "point_estimate", "point_estimate_text", "title")
    }
    if "remove_axis" in plot_kwargs:
        plot_kwargs_dist["remove_axis"] = plot_kwargs["remove_axis"]

    # print(f"\n aes_map = {aes_map}")

    # ---------STEP 1 (PPC data)-------------
    # print(f"\nposterior predictive distri = {pp_distribution!r}")

    # density calculation for observed variables
    pp_kwargs = copy(plot_kwargs.get("predictive", {}))
    # print(f"\nreduce_dims = {reduce_dims!r}")

    if pp_kwargs is not False:
        _, pp_density_aes, _ = filter_aes(plot_collection, aes_map, "predictive", reduce_dims)
        # print(f"\npp_density_aes = {pp_density_aes}\npp_density_ignore= {pp_density_ignore}")

        # getting first default color from color cycle and picking it
        pp_default_color = plot_bknd.get_default_aes("color", 1, {})[0]
        if "color" not in pp_density_aes:
            pp_kwargs.setdefault("color", pp_default_color)

        if "alpha" not in pp_density_aes:
            pp_kwargs.setdefault("alpha", 0.2)

        # passing plot_kwargs["predictive"] to plot_kwargs_dist (unlike tracedistplot there are
        # multiple artists generated via plot_dist in this plot- plot_ppc)
        plot_kwargs_dist[kind] = pp_kwargs

        # calling plot_dist with plot_collection and customized args
        pp_dt = DataTree(name="pp_dt", data=pp_distribution)  # has to be converted from a
        # dataarray to a datatree first before passing to plot_dist

        plot_dist(
            pp_dt,
            group="pp_dt",
            sample_dims=reduce_dims,
            kind=kind,
            plot_collection=plot_collection,
            labeller=labeller,
            aes_map={
                kind: aes_map.get("predictive", {})
            },  # aes_map[kind] is set to "predictive" aes_map
            plot_kwargs=plot_kwargs_dist,  # plot_kwargs[kind] is set to "predictive" plot_kwargs
            stats_kwargs={
                "density": stats_kwargs.get("predictive", {})
            },  # stats_kwargs["density"] is set to "predictive" stats_kwargs
            # via plot_dist
        )

    # ---------STEP 2 (PPC AGGREGATE)-------------

    if aggregate:  # all aggregate related logic happens here
        aggregate_kwargs = copy(plot_kwargs.get("aggregate", {}))

        _, aggregate_density_aes, _ = filter_aes(plot_collection, aes_map, "aggregate", reduce_dims)

        # print(
        #    f"\agg_dens_aes = {aggregate_density_aes}\nagg_dens_ignore= {aggregate_density_ignore}"
        # )

        if "linestyle" not in aggregate_density_aes:
            aggregate_kwargs.setdefault("linestyle", "--")

        # getting first two default colors from color cycle and picking the second
        aggregate_default_color = plot_bknd.get_default_aes("color", 2, {})[1]
        if "color" not in aggregate_density_aes:
            aggregate_kwargs.setdefault("color", aggregate_default_color)

        # print(f"\n aggregate reduce_dims = {reduce_dims!r}")
        # print(f"\n aggregate sample_dims = {sample_dims!r}")
        # print(f"\n aggregate_kwargs = {aggregate_kwargs}")
        aggregate_reduce_dims = reduce_dims + list(sample_dims)
        # print(f"\n aggregate_reduce_dims = {aggregate_reduce_dims}")

        # passing plot_kwargs["aggregate"] to plot_kwargs_dist
        plot_kwargs_dist[kind] = aggregate_kwargs

        plot_dist(
            pp_dt,
            group="pp_dt",
            sample_dims=aggregate_reduce_dims,
            kind=kind,
            plot_collection=plot_collection,
            labeller=labeller,
            aes_map={
                kind: aes_map.get("aggregate", {})
            },  # aes_map[kind] is set to "aggregate" aes_map
            plot_kwargs=plot_kwargs_dist,  # plot_kwargs[kind] is set to "aggregate" plot_kwargs
            stats_kwargs={
                "density": stats_kwargs.get("aggregate", {})
            },  # stats_kwargs["density"] is set to "aggregate" stats_kwargs
        )

    # ---------STEP 3 (observed data)-----------
    if observed:  # all observed data group plotting logic happens here
        observed_data_group = "observed_data"

        obs_distribution = process_group_variables_coords(
            dt,
            group=observed_data_group,
            var_names=var_names,
            filter_vars=filter_vars,
            coords=coords,
        )
        obs_kwargs = copy(plot_kwargs.get("observed", {}))

        _, obs_density_aes, _ = filter_aes(plot_collection, aes_map, "observed", reduce_dims)
        # print(f"\nobs_density_dims = {obs_density_dims}\nobs_density_aes = {obs_density_aes}")

        if "color" not in obs_density_aes:
            obs_kwargs.setdefault("color", "black")

        # print(f"\nobs_kwargs = {obs_kwargs}")

        # passing plot_kwargs["observed"] to plot_kwargs_dist
        plot_kwargs_dist[kind] = obs_kwargs

        obs_dt = DataTree(name="obs_dt", data=obs_distribution)
        plot_dist(
            obs_dt,
            group="obs_dt",
            sample_dims=reduce_dims,
            kind=kind,
            plot_collection=plot_collection,
            labeller=labeller,
            aes_map={
                kind: aes_map.get("observed", {})
            },  # aes_map[kind] is set to "observed" aes_map
            plot_kwargs=plot_kwargs_dist,  # plot_kwargs[kind] is set to "observed" plot_kwargs
            stats_kwargs={
                "density": stats_kwargs.get("observed", {})
            },  # stats_kwargs["density"] is set to "observed" stats_kwargs
        )

        # ---------STEP 4 (observed rug plot)-----------
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
