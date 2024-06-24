"""ppc plot code."""

# import warnings
from copy import copy
from numbers import Integral

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.sel_utils import xarray_sel_iter

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import labelled_title, line_xy


# define function
def plot_ppc(
    dt,
    kind=None,
    var_names=None,
    filter_vars=None,
    group="posterior",
    observed=None,
    coords=None,
    sample_dims=None,
    facet_dims=None,
    data_pairs=None,
    mean=True,
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
    kind : {"kde", "cumulative", "scatter"}, optional
        How to represent the marginal density. Defaults to ``rcParams["plot.density_kind"]``.
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
    facet_dims : list, optional
        Dimensions to facet over (for which multiple plots will be generated).
        Defaults to empty list.
    data_pairs : dict, optional
        Dictionary containing relations between observed data and posterior/prior predictive data.
        Dictionary structure:
            * key = observed data var_name
            * value = posterior/prior predictive var_name
        For example, data_pairs = {'y' : 'y_hat'}.
        If None, it will assume that the observed data and the posterior/prior predictive data
        have the same variable name
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

    # making sure both posterior/prior predictive group and observed_data group exists in
    # datatree provided
    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    for groups in (f"{group}_predictive", "observed_data"):
        if groups not in dt.children:
            raise TypeError(f'`data` argument must have the groups "{groups}" for ppcplot')

    # making sure kde type is one of these three
    if kind.lower() not in ("kde", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `cumulative`, or `scatter`")

    # initializaing data_pairs as empty dict in case pp and observed data var names are same
    if data_pairs is None:
        data_pairs = {}

    predictive_data_group = f"{group}_predictive"
    if observed is None:
        observed = group == "posterior"  # by default true if posterior, false if prior

    # process data to plot (select specified groups/variables/coords)
    # two distributions are created, one to hold pp observed data and one to hold actual
    # observed data

    # pp distribution group plotting logic
    pp_distribution = process_group_variables_coords(
        dt, group=predictive_data_group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

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
        pc_kwargs["aes"].setdefault("overlay", ["chain", "draw"])  # setting overlay dims
        plot_collection = PlotCollection.wrap(
            pp_distribution,
            backend=backend,
            **pc_kwargs,
        )
    reduce_dims = []
    reduce_dims.append(
        str(
            dim for dim in pp_distribution.dims if dim not in facet_dims.union(sample_dims)
        )  # set by elimination
    )
    reduce_dims = ["school"]

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("predictive", plot_collection.aes_set)
    if labeller is None:
        labeller = BaseLabeller()

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # checking plot_collection wrapped dataset and viz/aes datatrees
    print(f"\nplot_collection.data = {plot_collection.data}")
    print(f"\nplot_collection.aes = {plot_collection.aes}")
    print(f"\nplot_collection.viz = {plot_collection.viz}")

    # picking random pp dataset sample indexes
    # total_pp_samples = plot_collection.data.sizes["chain"] * plot_collection.data.sizes["draw"]
    total_pp_samples = np.prod(
        [plot_collection.data.sizes[dim] for dim in sample_dims if dim in plot_collection.data.dims]
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

    # Convert 1D indices to ND indices (where N=number of sample dims)
    sample_sizes = [pp_distribution.sizes[dim] for dim in sample_dims]
    sample_indices = np.unravel_index(pp_sample_ix, sample_sizes)
    print(f"\nUnravelled sample_indices = {sample_indices}")

    # Create an ND boolean mask
    mask = np.zeros(sample_sizes, dtype=bool)
    mask[sample_indices] = True

    # Convert the mask to an xarray DataArray
    mask = xr.DataArray(mask, dims=sample_dims)
    print(f"\nmask = {mask}")

    # Use the mask to subset pp_distribution
    pp_distribution = pp_distribution.where(mask, drop=True)

    # ---------STEP 1 (PPC data)-------------
    print(f"\nposterior predictive distri = {pp_distribution.obs!r}")

    # density calculation for observed variables
    pp_kwargs = copy(plot_kwargs.get("predictive", {}))
    print(f"\nreduce_dims = {reduce_dims!r}")

    if pp_kwargs is not False:
        pp_density_dims, pp_density_aes, pp_density_ignore = filter_aes(
            plot_collection, aes_map, "predictive", reduce_dims
        )
        print(f"\npp_density_dims = {pp_density_dims}\npp_density_aes= {pp_density_aes}")

        pp_density = pp_distribution.azstats.kde(
            dims=pp_density_dims, **stats_kwargs.get("predictive", {})
        )

        if "color" not in pp_density_aes:
            pp_kwargs.setdefault("color", "C0")

        if "alpha" not in pp_density_aes:
            pp_kwargs.setdefault("alpha", 0.2)

        print(f"\npp_density = {pp_density!r}")
        plot_collection.map(
            line_xy, "predictive", data=pp_density, ignore_aes=pp_density_ignore, **pp_kwargs
        )

    # ---------STEP 2 (observed data)-----------
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
            print(f"\nobs_density_dims = {pp_density_dims}\nobs_density_aes = {obs_density_aes}")

            obs_density = obs_distribution.azstats.kde(
                dims=obs_density_dims, **stats_kwargs.get("observed", {})
            )
            print(f"\nobserved data density = {obs_density}")

            if "color" not in obs_density_aes:
                obs_kwargs.setdefault("color", "black")

            obs_kwargs.setdefault("zorder", 3)

            plot_collection.map(
                line_xy, "observed", data=obs_density, ignore_aes=obs_density_ignore, **obs_kwargs
            )

    # ---------STEP 3 (PPC MEAN)------------- (WIP)

    mean_kwargs = copy(plot_kwargs.get("mean", {}))
    if mean and pp_kwargs is not False and mean_kwargs is not False:
        _, mean_density_aes, mean_density_ignore = filter_aes(
            plot_collection, aes_map, "mean", reduce_dims
        )
        pp_xs = []
        pp_densities = []

        subselections = xarray_sel_iter(pp_density, skip_dims=("plot_axis", "kde_dim"))
        # print(len(list(subselections)))
        for var, _, isel in subselections:
            x_values = pp_density[var].isel(**isel).sel(plot_axis="x")
            y_values = pp_density[var].isel(**isel).sel(plot_axis="y")
            if np.isnan(y_values.values).any():
                continue

            pp_xs.append(x_values.values)
            pp_densities.append(y_values.values)

        pp_xs = np.array(pp_xs)
        pp_densities = np.array(pp_densities)

        print(f"\npp_xs = {pp_xs!r}")
        print(f"\npp_densities = {pp_densities!r}")
        print(f"\nlength of pp_xs= {len(pp_xs)}")

        # mean calculation (from legacy arviz)
        rep = len(pp_densities)
        len_density = len(pp_densities[0])
        new_x = np.linspace(np.min(pp_xs), np.max(pp_xs), len_density)
        new_d = np.zeros((rep, len_density))
        bins = np.digitize(pp_xs, new_x, right=True)
        new_x -= (new_x[1] - new_x[0]) / 2
        for irep in range(rep):
            new_d[irep][bins[irep]] = pp_densities[irep]

        print(f"\nnew_x = {new_x!r}")
        print(f"\nnew_d = {new_d.mean(0)!r}")

        # initializing a mean density dataset to hold the computed mean data
        mean_density = xr.Dataset(
            {
                "obs": (("plot_axis", "kde_dim"), [new_x, new_d.mean(0)]),
            },
            coords={
                "plot_axis": ["x", "y"],
            },
        )

        print(f"\nmean_dataset = {mean_density!r}")

        if "linestyle" not in mean_density_aes:
            mean_kwargs.setdefault("linestyle", "--")

        if "color" not in mean_density_aes:
            mean_kwargs.setdefault("color", "C1")

        plot_collection.map(
            line_xy, "mean", data=mean_density, ignore_aes=mean_density_ignore, **mean_kwargs
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
