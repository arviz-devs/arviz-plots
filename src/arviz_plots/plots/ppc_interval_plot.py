"""Posterior predictive intervals plot."""

import numpy as np
import xarray as xr
from arviz_base import rcParams

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import ci_line_y, scatter_xy


def plot_ppc_intervals(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    point_estimate=None,
    ci_kind=None,
    ci_probs=None,
    x=None,
    plot_collection=None,
    backend=None,
    labeller=None,  # pylint: disable=unused-argument
    aes_by_visuals=None,
    visuals=None,
    **pc_kwargs,
):
    """Plot posterior predictive intervals with observed data overlaid.

    This plot is a posterior predictive check that helps to visualize how well the
    model's predictions capture the observed data. For each data point, it plots the
    credible interval of the posterior predictive distribution along with a point
    estimate (like the median) and the true observed value.

    Parameters
    ----------
    dt : DataTree
        Input data. It should contain the ``posterior_predictive`` and
        ``observed_data`` groups.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, "like", "regex"}, default=None
        If None, interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior_predictive"
        Group to be plotted.
    coords : dict, optional
        Coordinates of `var_names` to be plotted.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    point_estimate : {"mean", "median"}, optional
        Which point estimate to plot for the posterior predictive distribution.
        Defaults to rcParam ``stats.point_estimate``.
    ci_kind : {"hdi", "eti"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``.
    ci_probs : (float, float), optional
        Indicates the probabilities for the inner and outer credible intervals.
        Defaults to ``(0.5, rcParams["stats.ci_prob"])``.
    x : str, optional
        Coordinate variable to use for the x-axis. If None, the observation dimension
        coordinate is used.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str or False}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * ``outer_interval`` -> passed to :func:`~arviz_plots.visuals.ci_line_y`
        * ``inner_interval`` -> passed to :func:`~arviz_plots.visuals.ci_line_y`
        * ``point_estimate`` -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * ``observed`` -> passed to :func:`~arviz_plots.visuals.scatter_xy`

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Notes
    -----
    This plot shows several elements for each data point:
    * A thin outer credible interval (e.g., 94%)
    * A thicker inner credible interval (e.g., 50%)
    * A point estimate for the prediction (e.g., the median)
    * The true observed data point

    See Also
    --------
    plot_ppc_dist : Plot 1D marginals for the posterior/prior predictive and observed data.

    Examples
    --------
    Plot posterior predictive intervals for the radon dataset, with custom styling.

    .. plot::
        :context: close-figs

        >>> from arviz_base import load_arviz_data
        >>> import arviz_plots as azp
        >>>
        >>> azp.style.use("arviz-variat")
        >>> data = load_arviz_data("radon")
        >>> data_subset = data.isel(obs_id=range(50))
        >>> styling = {
        >>>     "outer_interval": {"width": 1.0, "color": "C0", "alpha": 0.5},
        >>>     "inner_interval": {"width": 2.5, "color": "C0"},
        >>>     "point_estimate": {"s": 20, "color": "C0"},
        >>>     "observed": {"s": 25, "marker": "o", "edgecolor": "black", "facecolor": "none"}
        >>> }
        >>> pc = azp.plot_ppc_intervals(
        >>>     data_subset,
        >>>     var_names=["y"],
        >>>     x="obs_id",
        >>>     visuals=styling,
        >>> )
    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"]
    if ci_kind not in ("hdi", "eti"):
        raise ValueError(f"ci_kind must be either 'hdi' or 'eti', but {ci_kind} was passed.")
    if point_estimate is None:
        point_estimate = rcParams["stats.point_estimate"]
    if ci_probs is None:
        rc_ci_prob = rcParams["stats.ci_prob"]
        ci_probs = (0.5, rc_ci_prob)

    ci_probs = np.array(ci_probs)
    if ci_probs.size != 2:
        raise ValueError("ci_probs must have two elements for inner and outer intervals.")
    if np.any(ci_probs < 0) or np.any(ci_probs > 1):
        raise ValueError("ci_probs must be between 0 and 1.")
    ci_probs.sort()

    predictive_dist = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    observed_dist = process_group_variables_coords(
        dt, group="observed_data", var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    obs_dims = [dim for dim in observed_dist.dims if dim not in sample_dims]
    if len(obs_dims) != 1:
        raise ValueError(f"Could not determine observation dimension from {obs_dims}.")
    obs_dim_name = obs_dims[0]

    if x is not None:
        x_data = observed_dist[x]
    else:
        x_data = observed_dist[obs_dim_name]

    if plot_collection is None:
        pc_kwargs = {"aes": {"x": x_data.dims, "y": observed_dist.dims}}
        plot_collection = PlotCollection.grid(
            observed_dist,
            backend=backend,
            **pc_kwargs,
        )

    if ci_kind == "eti":
        ci_fun = predictive_dist.azstats.eti
    elif ci_kind == "hdi":
        ci_fun = predictive_dist.azstats.hdi

    outer_interval = ci_fun(prob=ci_probs[1], dim=sample_dims)
    inner_interval = ci_fun(prob=ci_probs[0], dim=sample_dims)

    if point_estimate == "median":
        point = predictive_dist.median(dim=sample_dims)
    elif point_estimate == "mean":
        point = predictive_dist.mean(dim=sample_dims)
    else:
        raise ValueError(
            f"point_estimate must be 'mean' or 'median', but {point_estimate} was passed."
        )

    visuals = {} if visuals is None else visuals
    aes_by_visuals = {} if aes_by_visuals is None else aes_by_visuals

    _, _, ci_ignore = filter_aes(plot_collection, aes_by_visuals, "credible_interval", sample_dims)

    for var_name in predictive_dist.data_vars:
        print(f"\n[DEBUG] Preparing to plot variable: '{var_name}'")

        var_point = point[var_name]
        var_obs = observed_dist[var_name]
        var_outer = outer_interval[var_name]
        var_inner = inner_interval[var_name]

        outer_interval_for_ci_line = xr.concat(
            [
                var_outer.sel(ci_bound="lower").reset_coords("ci_bound", drop=True),
                var_outer.sel(ci_bound="upper").reset_coords("ci_bound", drop=True),
                x_data,
            ],
            dim="plot_axis",
        ).assign_coords(plot_axis=["y_bottom", "y_top", "x"])

        inner_interval_for_ci_line = xr.concat(
            [
                var_inner.sel(ci_bound="lower").reset_coords("ci_bound", drop=True),
                var_inner.sel(ci_bound="upper").reset_coords("ci_bound", drop=True),
                x_data,
            ],
            dim="plot_axis",
        ).assign_coords(plot_axis=["y_bottom", "y_top", "x"])

        point_for_scatter = xr.concat([var_point, x_data], dim="plot_axis").assign_coords(
            plot_axis=["y", "x"]
        )
        observed_for_scatter = xr.concat([var_obs, x_data], dim="plot_axis").assign_coords(
            plot_axis=["y", "x"]
        )

        outer_kwargs = visuals.get("outer_interval", {}).copy()
        plot_collection.map(
            ci_line_y,
            f"{var_name}_outer_interval",
            data=outer_interval_for_ci_line.rename(var_name),
            ignore_aes=ci_ignore,
            **outer_kwargs,
        )

        inner_kwargs = visuals.get("inner_interval", {}).copy()
        inner_kwargs.setdefault("width", 3)
        plot_collection.map(
            ci_line_y,
            f"{var_name}_inner_interval",
            data=inner_interval_for_ci_line.rename(var_name),
            ignore_aes=ci_ignore,
            **inner_kwargs,
        )

        point_kwargs = visuals.get("point_estimate", {}).copy()
        plot_collection.map(
            scatter_xy,
            f"{var_name}_point_estimate",
            data=point_for_scatter.rename(var_name),
            ignore_aes={"x", "y"},
            **point_kwargs,
        )

        observed_kwargs = visuals.get("observed", {}).copy()
        observed_kwargs.setdefault("color", "black")
        plot_collection.map(
            scatter_xy,
            f"{var_name}_observed",
            data=observed_for_scatter.rename(var_name),
            ignore_aes={"x", "y"},
            **observed_kwargs,
        )

    return plot_collection
