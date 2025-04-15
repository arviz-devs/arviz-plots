from arviz_base import rcParams
from importlib import import_module
from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import set_figure_layout, process_group_variables_coords, filter_aes
from arviz_base.labels import BaseLabeller


def plot_ppc_tstat(
    dt,
    group="posterior_predictive",
    t_stat="median",
    var_names=None,
    filter_vars=None,
    sample_dims=None,
    plot_collection=None,
    coords=None,
    backend=None,
    data_pairs=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """
    Plot Bayesian t-stat for the posterior/prior predictive distribution.

    Parameters
    ----------
    dt : DataTree
        Input data
    group : str,
        Group to be plotted. Defaults to "posterior_predictive".
        It could also be "prior_predictive".
    t_stat : str, float, default "median"
        Test statistics to compute from the observations and predictive distributions.
        Allowed strings are “mean”, “median” or “std”. Alternative a quantile can be passed
        as a float (or str) in the interval (0, 1).
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default=None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    coords : dict, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    data_pairs : dict, optional
        Dictionary of keys prior/posterior predictive data and values observed data variable names.
        If None, it will assume that the observed data and the predictive data have
        the same variable name.
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.
    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

            * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
            * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

    stats_kwargs : mapping, optional
        Valid keys are:

        * density -> passed to kde.

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    """
    if group not in ("posterior_predictive", "prior_predictive"):
        raise TypeError("`group` argument must be either `posterior_predictive` or `prior_predictive`")
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if stats_kwargs is None:
        stats_kwargs = {}
    else:
        stats_kwargs = stats_kwargs.copy()
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()
    if labeller is None:
        labeller = BaseLabeller()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    if data_pairs is None:
        data_pairs = (var_names, var_names)
    else:
        data_pairs = (list(data_pairs.keys()), list(data_pairs.values()))

    predictive_dist = process_group_variables_coords(
        dt, group=group, var_names=data_pairs[0], filter_vars=filter_vars, coords=coords
    )

    predictive_dist = predictive_dist.stack(sample=sample_dims)
    if t_stat == "median":
        predictive_dist = predictive_dist.median(dim=list(predictive_dist.dims)[0])
    elif t_stat == "mean":
        predictive_dist = predictive_dist.mean(dim=list(predictive_dist.dims)[0])
    elif t_stat == "std":
        predictive_dist = predictive_dist.std(dim=list(predictive_dist.dims)[0])
    else:
        try:
            t_stat_float = float(t_stat)
        except ValueError:
            raise ValueError(f"T statistics '{t_stat}' not implemented")
        if 0 < t_stat_float < 1:
            return predictive_dist.quantile(q=t_stat_float, dim=list(predictive_dist.dims)[0])
        else:
            raise ValueError(f"T statistic '{t_stat}' not in valid range (0, 1).")
    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs.setdefault("rows", None)

        pc_kwargs = set_figure_layout(pc_kwargs, plot_bknd, predictive_dist)

        plot_collection = PlotCollection.wrap(
            predictive_dist,
            backend=backend,
            **pc_kwargs,
        )
    # Plot KDE lines
    plot_dist(
        predictive_dist,
        var_names=None,
        group=None,
        coords=None,
        sample_dims=["sample"],
        point_estimate=None,
        ci_kind=None,
        ci_prob=None,
        plot_collection=plot_collection,
        aes_map=aes_map,
        backend=backend,
        labeller=labeller,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
        pc_kwargs=pc_kwargs
    )

    return plot_collection
