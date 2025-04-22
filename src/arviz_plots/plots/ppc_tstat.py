"""ppc t-stat plot code."""
from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import process_group_variables_coords, set_wrap_layout
from arviz_plots.visuals import scatter_x


def plot_ppc_tstat(
    dt,
    var_names=None,
    group="posterior_predictive",
    filter_vars=None,
    sample_dims=None,
    t_stat="median",
    kind=None,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
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
    Plot Bayesian t-stat for observed data and posterior/prior predictive.

    Parameters
    ----------
    dt : DataTree
        Input data
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    group : str,
        Group to be plotted. Defaults to "posterior_predictive".
        It could also be "prior_predictive".
    filter_vars : {None, “like”, “regex”}, default=None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    t_stat : str, float, or callable() default "median"
        Test statistics to compute from the observations and predictive distributions.
        Allowed strings are “mean”, “median”, “std”, “var”, “min”, “max”, “iqr”
        (interquartile range) and “mad” (median absolute deviation). Alternative a
        quantile can be passed as a float (or str) in the interval (0, 1). Finally,
        a user defined function is also accepted.
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to rcParam :data:`stats.point_estimate`
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to ``rcParams["stats.ci_prob"]``
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

        * One of "kde", "ecdf", "dot" or "hist", matching the `kind` argument.

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * observed_tstat -> passed to :func:`~arviz_plots.visuals.scatter_x`.
        * credible_interval -> passed to :func:`~arviz_plots.visuals.line_x`. Defaults to False.
        * point_estimate -> passed to :func:`~arviz_plots.visuals.scatter_x`. Defaults to False.
        * point_estimate_text -> passed to :func:`~arviz_plots.visuals.point_estimate_text`.
          Defaults to False.
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * rug -> passed to :func:`~arviz_plots.visuals.scatter_x`. Defaults to False.
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats_kwargs : mapping, optional
        Valid keys are:

        * density -> passed to kde, ecdf, ...

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection


    Examples
    --------
    Use 25th percentile (quantile 0.25) as t-statistic

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_tstat, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('radon')
        >>> plot_ppc_tstat(dt, t_stat="0.25")

    Define custom t-statistic function and plot histogram

    .. plot::
        :context: close-figs

        >>> def cv(x):
        >>>     return np.std(x, axis=0) / np.mean(x, axis=0)
        >>> plot_ppc_tstat(dt, t_stat=cv, kind="hist")


    Use median as t-statistic and plot point-interval

    .. plot::
        :context: close-figs

        >>> azp.plot_ppc_tstat(dt,
        >>>                    plot_kwargs={"kde": False,
        >>>                                 "credible_interval": {},
        >>>                                 "point_estimate": {}})


    .. minigallery:: plot_ppc_tstat


    """
    if group not in ("posterior_predictive", "prior_predictive"):
        raise TypeError(
            "`group` argument must be either `posterior_predictive` or `prior_predictive`"
        )
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

    observed_dist = process_group_variables_coords(
        dt, group="observed_data", var_names=data_pairs[1], filter_vars=filter_vars, coords=coords
    )

    predictive_dist = predictive_dist.stack(sample=sample_dims)
    reduce_dim = [dim for dim in predictive_dist.dims if dim != "sample"]
    if t_stat in ["mean", "median", "std", "var", "min", "max"]:
        predictive_dist = getattr(predictive_dist, t_stat)(dim=reduce_dim)
        observed_dist = getattr(observed_dist, t_stat)()
        plot_kwargs.setdefault("title", {"text": t_stat})
    elif t_stat == "iqr":

        def iqr(data, dim):
            q25 = data.quantile(q=0.25, dim=dim)
            q75 = data.quantile(q=0.75, dim=dim)
            return q75 - q25

        predictive_dist = iqr(predictive_dist, dim=reduce_dim)
        observed_dist = iqr(observed_dist, dim=None)
        plot_kwargs.setdefault("title", {"text": "IQR"})
    elif t_stat == "mad":

        def mad(data, dim):
            median = data.median(dim=dim)
            return np.abs((data - median)).median(dim=dim)

        predictive_dist = mad(predictive_dist, dim=reduce_dim)
        observed_dist = mad(observed_dist, dim=None)
        plot_kwargs.setdefault("title", {"text": "MAD"})

    elif hasattr(t_stat, "__call__"):
        predictive_dist = predictive_dist.map(t_stat)
        observed_dist = observed_dist.map(t_stat)
        plot_kwargs.setdefault("title", {"text": t_stat.__name__})
    else:
        try:
            t_stat_float = float(t_stat)
        except ValueError as ve:
            raise ValueError(f"T statistics '{t_stat}' not implemented") from ve
        if 0 < t_stat_float < 1:
            predictive_dist = predictive_dist.quantile(q=t_stat_float, dim=reduce_dim).rename(
                {"quantile": "t_stat"}
            )
            observed_dist = observed_dist.quantile(q=t_stat_float).rename({"quantile": "t_stat"})
            plot_kwargs.setdefault("title", {"text": f"q={t_stat}"})
        else:
            raise ValueError(f"T statistic '{t_stat}' not in valid range (0, 1).")

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs.setdefault("rows", None)

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, predictive_dist)

        plot_collection = PlotCollection.wrap(
            predictive_dist,
            backend=backend,
            **pc_kwargs,
        )
    # Plot predictive data
    plot_kwargs.setdefault("credible_interval", False)
    plot_kwargs.setdefault("point_estimate", False)
    plot_kwargs.setdefault("point_estimate_text", False)

    plot_dist(
        predictive_dist,
        var_names=None,
        group=None,
        coords=None,
        sample_dims=["sample"],
        kind=kind,
        point_estimate=point_estimate,
        ci_kind=ci_kind,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        aes_map=aes_map,
        backend=backend,
        labeller=labeller,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
        pc_kwargs=pc_kwargs,
    )

    # Plot the observed data
    observed_tstat_kwargs = copy(plot_kwargs.get("observed_tstat", {}))
    if observed_tstat_kwargs is not False:
        observed_tstat_kwargs.setdefault("color", "black")
        plot_collection.map(
            scatter_x, "observed_tstat", data=observed_dist.mean(), **observed_tstat_kwargs
        )

    return plot_collection
