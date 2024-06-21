"""ppc plot code."""

# import warnings
from copy import copy
from numbers import Integral

import arviz_stats  # pylint: disable=unused-import
import numpy as np

# import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection, concat_model_dict
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import line_xy


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
    data_pairs=None,
    # mean=True,
    flatten=None,
    flatten_pp=None,
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
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to
        aesthetics.
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
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "cumulative", "scatter"}, optional
        How to represent the marginal density. Defaults to ``rcParams["plot.density_kind"]``.
    data_pairs : dict, optional
        Dictionary containing relations between observed data and posterior/prior predictive data.
        Dictionary structure:
            * key = observed data var_name
            * value = posterior/prior predictive var_name
        For example, data_pairs = {'y' : 'y_hat'}.
        If None, it will assume that the observed data and the posterior/prior predictive data
        have the same variable name
    flatten : list of str, optional
        Dimensions to flatten in the observed_data.
        Only flattens across the coordinates specified in the coords argument.
        Defaults to flattening all of the dimensions.
    flatten_pp : list of str, optional
        Dimensions to flatten in the posterior predictive data.
        Only flattens across the coordinates specified in the coords argument.
        Defaults to flattening all of the dimensions.
        Dimensions should match flatten excluding dimensions for data_pairs parameters.
        If flatten is defined and flatten_pp is None, then flatten_pp = flatten.
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

    # making sure both posterior/prior predictive group and observed_data group exists in
    # datatree provided
    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    for groups in (f"{group}_predictive", "observed_data"):
        if not hasattr(dt, groups):
            raise TypeError(f'`data` argument must have the group "{groups}" for ppcplot')

    # making sure kde type is one of these three
    if kind.lower() not in ("kde", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `cumulative`, or `scatter`")

    # initializaing data_pairs as empty dict in case pp and observed data var names are same
    if data_pairs is None:
        data_pairs = {}

    if group == "posterior":
        predictive_data_group = "posterior_predictive"
        if observed is None:
            observed = True
    elif group == "prior":
        predictive_data_group = "prior_predictive"
        if observed is None:
            observed = False

    if observed:
        observed_data_group = "observed_data"

    # process data to plot (select specified groups/variables/coords)
    # two distributions are created, one to hold pp observed data and one to hold actual
    # observed data

    obs_distribution = process_group_variables_coords(
        dt, group=observed_data_group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if flatten is None:
        flatten = list(
            obs_distribution.dims
        )  # assigning all dims to flatten in case user does not provide specific ones

    pp_distribution = process_group_variables_coords(
        dt, group=predictive_data_group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if flatten_pp is None:
        flatten_pp = flatten

    # concatenate both distributions into one or just put them into a dict to pass to .wrap() which
    # will add
    # them along the new model dimension
    distribution = {"posterior_predictive": pp_distribution, "observed_data": obs_distribution}
    distribution = concat_model_dict(
        distribution
    )  # converts into a single dataset along new 'model' dim
    print(
        f"plot_ppc merged distribution= {distribution!r}"
    )  # after concatenating, only one variable "obs" exists
    print(f"plot_ppc distribution obs variable values= {distribution.obs.values}")

    # an advantage of having one variable is that .wrap() will not cause process_facet_dims to
    # create separate plots for each variable, but if multiple subplots are wanted (for example
    # for each coord of a dimension not to be flattened) then that can still be done by adjusting
    # pc_kwargs

    # facetting overall isnt very important for plot_ppc though since usually by default there'll
    # be only one plot multiple plots would only matter in the case of non sample dims that
    # have coords

    # wip: dims selected to be flattened should be dealt with before wrapping or maybe have
    # aesthetics set for them automatically/as-a-requirement if not to be flattened?

    # wrap plot collection
    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]  # special variable to create one plot per variable
            + [
                dim for dim in distribution.dims if dim not in {"model"}.union(distribution.dims)
            ],  # zero dims are selected here
        )  # for plot_ppc(), selecting all dims to reduce by default, and not just sample dims
        # (chain,draw) like plot_dist
        # ^this is because plot_ppc() default is just one plot, though users can separate by a
        # dim's coords should they wish explicitly
        if "model" in distribution:
            pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
            pc_kwargs["aes"].setdefault("color", ["model"])
            pc_kwargs["aes"].setdefault("y", ["model"])
        # process_facet_dims is called within wrap() to create the subplotting areas- just 1 by
        # default
        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    # aes_map.setdefault(kind, plot_collection.aes_set.difference("y"))
    if labeller is None:
        labeller = BaseLabeller()

    if random_seed is not None:
        np.random.seed(random_seed)

    # checking plot_collection wrapped dataset and viz/aes datatrees
    print(f"\nplot_collection.data = {plot_collection.data}")
    print(f"\nplot_collection.aes = {plot_collection.aes}")
    print(f"\nplot_collection.viz = {plot_collection.viz}")

    # picking random pp dataset sample indexes
    total_pp_samples = plot_collection.data.sizes["chain"] * plot_collection.data.sizes["draw"]
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

    pp_sample_ix = np.random.choice(total_pp_samples, size=num_pp_samples, replace=False)

    print(f"\npp_sample_ix: {pp_sample_ix!r}")

    # iterate over ppc/observed data, subsetting and doing statistical computation of kde as
    # required

    # data structure implementation: could divide the ppc obs variable into coords along a
    # new 'ppc_dim' dimension for each pp_sample and make sure mapping of each is done

    # ---------STEP 1 (observed data)-----------
    # subset the distribution data for observed data model, calculate density and call map()
    observed_distribution = distribution.sel(model="observed_data")
    print(
        f"\nobserved distri = {observed_distribution.obs!r}"
    )  # the obs data was auto broadcasted all over the chain and draw dims
    # but we can ignore the other dims and just subset it to 1 chain and 1 draw and then use the
    # resulting subsetted variable data
    observed_distribution = observed_distribution.sel(chain=0, draw=0)
    print(f"\nobserved distri = {observed_distribution.obs!r}")

    # density calculation for observed variables
    density_kwargs = copy(plot_kwargs.get(kind, {}))

    if density_kwargs is not False:
        density_dims, _, density_ignore = filter_aes(plot_collection, aes_map, "kde", sample_dims)
        print(f"\ndensity_dims = {density_dims}\ndensity_ignore= {density_ignore}")

    for dim in flatten:  # flatten is the list of user defined dims to flatten or all dims
        obs_density_dims = {dim}.union(
            density_dims
        )  # dims to be reduced now includes the flatten ones and not just sample dims

    obs_density = observed_distribution.azstats.kde(
        dims=obs_density_dims, **stats_kwargs.get("density", {})
    )
    print(f"\nobserved data density = {obs_density}")

    plot_collection.map(
        line_xy, "kde", data=obs_density, ignore_aes=density_ignore, **density_kwargs
    )

    # ---------STEP 2 (PPC data)-------------
    # subset distribution for predictive data model, reshape and pick samples and flatten and
    # then call kde to get density info then call map() again with this density info
    # (algorithm of legacy plot ppc followed for this, but implemented in refactored way)
    predictive_distribution = distribution.sel(model="posterior_predictive")
    print(f"\nposterior predictive distri = {predictive_distribution.obs!r}")

    predictive_distribution = predictive_distribution.stack(ppc_dim=("chain", "draw"))  # reshaping
    predictive_distribution = predictive_distribution.assign_coords(
        ppc_dim=np.arange(predictive_distribution.sizes["ppc_dim"])
    )
    print(f"\nposterior predictive distri reshaped = {predictive_distribution!r}")
    print(f"\nposterior predictive distri stacked variable = {predictive_distribution.obs!r}")

    # selecting sampled values from predictive_distribution only
    predictive_distribution = predictive_distribution.isel(ppc_dim=pp_sample_ix)
    print(f"\nposterior predictive distri selected samples = {predictive_distribution.obs!r}")

    for dim in flatten_pp:  # flatten is the list of user defined dims to flatten or all dims
        pp_density_dims = {dim}.union(
            density_dims
        )  # dims to be reduced now includes the flatten ones and not just sample dims
    # ppc_dim should not be flattened- this is because subselections of this correspond to
    # pp samples

    pp_densities = []
    pp_xs = []

    for i in range(predictive_distribution.sizes["ppc_dim"]):
        # select the i-th subselection along 'ppc_dim'
        subselection = predictive_distribution.isel(ppc_dim=i)

        # compute the density of the subselection
        pp_density = subselection.azstats.kde(
            dims=pp_density_dims, **stats_kwargs.get("density", {})
        )
        print(f"\npredictive data density for subselection {i} = {pp_density}")
        pp_densities.append(pp_density.sel(plot_axis="y").values)
        pp_xs.append(pp_density.sel(plot_axis="x").values)  # storing these for later mean calc

        plot_collection.map(
            line_xy, "kde", data=pp_density, ignore_aes=density_ignore, **density_kwargs
        )

    print(f"\npp_densities= {pp_densities}")
    print(f"\npp_xs = {pp_xs}")

    # ---------STEP 3 (PPC MEAN)------------- (WIP)

    # checking plot_collection wrapped dataset and viz/aes datatrees
    print("\nAfter .map() of density as kde artist")
    print(f"\nplot_collection.data = {plot_collection.data}")
    print(f"\nplot_collection.aes = {plot_collection.aes}")
    print(f"\nplot_collection.viz = {plot_collection.viz}")

    print("End of plot_ppc()")
    return plot_collection
