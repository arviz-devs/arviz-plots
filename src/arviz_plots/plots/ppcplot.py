"""ppc plot code."""

import warnings
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
from arviz_plots.visuals import labelled_title


# define function
def plot_ppc(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    observed=None,
    coords=None,
    sample_dims=None,
    kind=None,
    facet_dims=None,
    data_pairs=None,
    aggregate=True,
    num_pp_samples=None,
    random_seed=None,
    # jitter=None,
    animated=False,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot prior/posterior predictive and observed values as kde, cumulative or scatter plots.

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
    coords : dict, optional
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for that dimension.
        Defaults to including all coordinates for all dimensions if None.
    sample_dims : str or sequence of hashable, optional
        Dimensions to loop over in plotting posterior/prior predictive.
        Note: Dims not in sample_dims or facet_dims (below) will be reduced by default.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "cumulative", "scatter"}, optional
        How to represent the marginal density. Defaults to ``rcParams["plot.density_kind"]``.
    facet_dims : list, optional
        Dimensions to facet over (for which multiple plots will be generated).
        Defaults to empty list. A warning is raised if `pc_kwargs` is also used to define
        dims to facet over with the `cols` key and `pc_kwargs` takes precedence.
    data_pairs : dict, optional
        Dictionary containing relations between observed data and posterior/prior predictive data.
        Dictionary structure:
            * key = observed data var_name
            * value = posterior/prior predictive var_name
        For example, data_pairs = {'y' : 'y_hat'}.
        If None, it will assume that the observed data and the posterior/prior predictive data
        have the same variable name
    aggregate: bool, optional
        If True, predictive data will be aggregated over both sample_dims and reduce_dims
    num_pp_samples : int, optional
        Number of prior/posterior predictive samples to plot.
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

        * One of "kde", "cumulative", "scatter", matching the `kind` argument

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "cumulative" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "scatter" -> passed to :func: `~arviz_plots.visuals.scatter_x`

        * "title" -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * "remove_axis" -> not passed anywhere, can only be ``False`` to skip calling this function

    stats_kwargs : mapping, optional
        Valid keys are:

        * density -> passed to kde, cumulative, ...

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
        duplicated_dims = set(facet_dims).intersection(pc_kwargs["cols"])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=f"""Facet dimensions have been defined twice.
                Both in the the top level function arguments and in `pc_kwargs`. 
                The `cols` key in `pc_kwargs` will take precedence.
                                        
                facet_dims = {facet_dims}
                pc_kwargs["cols"] = {pc_kwargs["cols"]}
                Duplicated dimensions: {duplicated_dims}.""",
            )
        # setting facet_dims to pc_kwargs defined values since it is used
        # later to calculate dims to reduce
        facet_dims = list(set(pc_kwargs["col"]).difference({"__variable__"}))

    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    predictive_data_group = f"{group}_predictive"
    if observed is None:
        observed = group == "posterior"  # by default true if posterior, false if prior

    # checking if plot_kwargs["observed"] or plot_kwargs["aggregate"] is not inconsistent with
    # top level bool arguments `observed`` and `aggregate`
    if "observed" in plot_kwargs:
        if plot_kwargs["observed"] != observed:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="""`plot_kwargs['observed']` inconsistency detected.
                    It is not the same as with the top level `observed` argument. 
                    `plot_kwargs['observed']` will take precedence.""",
                )
            observed = plot_kwargs["observed"]

    # making sure both posterior/prior predictive group and observed_data group exists in
    # datatree provided
    if observed:
        for group_name in (f"{predictive_data_group}", "observed_data"):
            if group_name not in dt.children:
                raise TypeError(f'`data` argument must have the group "{group_name}" for ppcplot')
    else:
        if f"{predictive_data_group}" not in dt.children:
            raise TypeError(f'`data` argument must have the group "{group}_predictive" for ppcplot')

    # making sure kde type is one of these three
    if kind.lower() not in ("kde", "ecdf", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `ecdf`, or `scatter`")

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
        dt, group=predictive_data_group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    # creating random number generator
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # picking random pp dataset sample indexes and subsetting pp_distribution accordingly
    # total_pp_samples = plot_collection.data.sizes["chain"] * plot_collection.data.sizes["draw"]
    total_pp_samples = np.prod(
        [pp_distribution.sizes[dim] for dim in sample_dims if dim in pp_distribution.dims]
    )
    if num_pp_samples is None:
        if kind == "scatter" and not animated:
            num_pp_samples = min(5, total_pp_samples)
        else:
            num_pp_samples = total_pp_samples

    if (
        not isinstance(num_pp_samples, Integral)
        or num_pp_samples < 1
        or num_pp_samples > total_pp_samples
    ):
        raise TypeError(f"`num_pp_samples` must be an integer between 1 and {total_pp_samples}.")

    pp_sample_ix = rng.choice(total_pp_samples, size=num_pp_samples, replace=False)

    print(f"\npp_sample_ix: {pp_sample_ix!r}")

    # stacking sample dims into a new 'ppc_dim' dimension
    pp_distribution = pp_distribution.stack(ppc_dim=sample_dims)

    # Select the desired samples
    pp_distribution = pp_distribution.isel(ppc_dim=pp_sample_ix)

    # renaming sample_dims so that rest of plot will consider this as sample_dims
    sample_dims = ["ppc_dim"]

    # wrap plot collection with pp distribution
    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
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
    print(f"\nreduce_dims={reduce_dims}")

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("predictive", plot_collection.aes_set)
    if labeller is None:
        labeller = BaseLabeller()

    # checking plot_collection wrapped dataset and viz/aes datatrees
    print(f"\nplot_collection.data = {plot_collection.data}")
    print(f"\nplot_collection.aes = {plot_collection.aes}")
    print(f"\nplot_collection.viz = {plot_collection.viz}")

    # ---------STEP 1 (PPC data)-------------
    print(f"\nposterior predictive distri = {pp_distribution!r}")

    # density calculation for observed variables
    pp_kwargs = copy(plot_kwargs.get("predictive", {}))
    print(f"\nreduce_dims = {reduce_dims!r}")

    if pp_kwargs is not False:
        _, pp_density_aes, pp_density_ignore = filter_aes(
            plot_collection, aes_map, "predictive", reduce_dims
        )
        print(f"\npp_density_aes = {pp_density_aes}\npp_density_ignore= {pp_density_ignore}")

        # getting first default color from color cycle and picking it
        pp_default_color = plot_bknd.get_default_aes("color", 1, {})[0]
        if "color" not in pp_density_aes:
            pp_kwargs.setdefault("color", pp_default_color)

        if "alpha" not in pp_density_aes:
            pp_kwargs.setdefault("alpha", 0.2)

        plot_dist_artists = ("credible_interval", "point_estimate", "point_estimate_text", "title")

        print(f"\npp_kwargs = {pp_kwargs}")

        # calling plot_dist with plot_collection and customized args
        pp_dt = DataTree(name="pp_dt", data=pp_distribution)  # has to be converted from a
        # dataarray to a datatree first before passing to plot_dist
        plot_dist(
            pp_dt,
            var_names=var_names,
            filter_vars=filter_vars,
            group="pp_dt",
            coords=coords,
            sample_dims=reduce_dims,
            kind=kind,
            plot_collection=plot_collection,
            labeller=labeller,
            aes_map={
                kind: value
                for key, value in aes_map.items()
                if key == "predictive" and key not in pp_density_ignore
            },
            plot_kwargs={
                **{key: False for key in plot_dist_artists},
                kind: dict(pp_kwargs.items()),
            },
            stats_kwargs={key: value for key, value in stats_kwargs.items() if key == "predictive"},
        )

    # ---------STEP 2 (PPC AGGREGATE)-------------

    aggregate_kwargs = copy(plot_kwargs.get("aggregate", {}))
    if aggregate and pp_kwargs is not False and aggregate_kwargs is not False:
        _, aggregate_density_aes, aggregate_density_ignore = filter_aes(
            plot_collection, aes_map, "aggregate", reduce_dims
        )

        print(
            f"\agg_dens_aes = {aggregate_density_aes}\nagg_dens_ignore= {aggregate_density_ignore}"
        )

        if "linestyle" not in aggregate_density_aes:
            aggregate_kwargs.setdefault("linestyle", "--")

        # getting first two default colors from color cycle and picking the second
        aggregate_default_color = plot_bknd.get_default_aes("color", 2, {})[1]
        if "color" not in aggregate_density_aes:
            aggregate_kwargs.setdefault("color", aggregate_default_color)

        print(f"\n aggregate reduce_dims = {reduce_dims!r}")
        print(f"\n aggregate sample_dims = {sample_dims!r}")
        print(f"\n aggregate_kwargs = {aggregate_kwargs}")
        aggregate_reduce_dims = reduce_dims + list(sample_dims)
        print(f"\n aggregate_reduce_dims = {aggregate_reduce_dims}")

        # setting aggregate aes_map to `[]` so `overlay` isn't applied
        aes_map.setdefault("aggregate", [])
        print(f"aes_map = {aes_map}")

        plot_dist(
            pp_dt,
            var_names=var_names,
            filter_vars=filter_vars,
            group="pp_dt",
            coords=coords,
            sample_dims=aggregate_reduce_dims,
            kind=kind,
            plot_collection=plot_collection,
            labeller=labeller,
            aes_map={
                kind: value  # aes_map["aggregate"] value gets assigned to kind
                for key, value in aes_map.items()
                if key == "aggregate" and key not in aggregate_density_ignore
            },
            plot_kwargs={
                **{key: False for key in plot_dist_artists},
                kind: dict(aggregate_kwargs.items()),
            },
            stats_kwargs={key: value for key, value in stats_kwargs.items() if key == "aggregate"},
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

        if obs_kwargs is not False:
            obs_density_dims, obs_density_aes, obs_density_ignore = filter_aes(
                plot_collection, aes_map, "observed", reduce_dims
            )
            print(f"\nobs_density_dims = {obs_density_dims}\nobs_density_aes = {obs_density_aes}")

            if "color" not in obs_density_aes:
                obs_kwargs.setdefault("color", "black")

            plot_dist_artists = (
                "credible_interval",
                "point_estimate",
                "point_estimate_text",
                "title",
            )

            print(f"\nobs_kwargs = {obs_kwargs}")

            obs_dt = DataTree(name="obs_dt", data=obs_distribution)
            plot_dist(
                obs_dt,
                var_names=var_names,
                filter_vars=filter_vars,
                group="obs_dt",
                coords=coords,
                sample_dims=reduce_dims,
                kind=kind,
                plot_collection=plot_collection,
                labeller=labeller,
                aes_map={
                    kind: value
                    for key, value in aes_map.items()
                    if key == "observed" and key not in obs_density_ignore
                },
                plot_kwargs={
                    **{key: False for key in plot_dist_artists},
                    "kde": dict(obs_kwargs.items()),
                },
                stats_kwargs={
                    kind: value for key, value in stats_kwargs.items() if key == "observed"
                },
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

    # checking plot_collection wrapped dataset and viz/aes datatrees
    print("\nAfter .map() of density as kde artist")
    print(f"\nplot_collection.data = {plot_collection.data}")
    print(f"\nplot_collection.aes = {plot_collection.aes}")
    print(f"\nplot_collection.viz = {plot_collection.viz}")

    print(f"\nsample_dims = {sample_dims}")
    print(f"\nfacet_dims = {facet_dims}")
    print(f"\nreduce_dims = {reduce_dims}")
    print("End of plot_ppc()")

    return plot_collection
