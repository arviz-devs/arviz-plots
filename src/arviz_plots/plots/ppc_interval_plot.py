"""Predictive intervals plot."""

from importlib import import_module
import numpy as np

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import get_contrast_colors, get_visual_kwargs, filter_aes, process_group_variables_coords, set_wrap_layout
from arviz_plots.visuals import ci_bound_y,  point_y


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

    Displays each observed value together with two credible intervals of the predictive distribution
    and a point estimate.

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
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot for the posterior predictive distribution.
        Defaults to rcParam ``stats.point_estimate``.
    ci_kind : {"hdi", "eti"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``.
    ci_probs : (float, float), optional
        Indicates the probabilities for the inner and outer credible intervals.
        Defaults to ``(0.5, rcParams["stats.ci_prob"])``. It's assumed that
        ``ci_probs[0] < ci_probs[1]``, otherwise they are sorted.
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

        * trunk, twig -> passed to :func:`~.visuals.line_x`
        * predictive_markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * observed_markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`.
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    See Also
    --------
    plot_ppc_dist : Plot 1D marginals for the posterior/prior predictive and observed data.
    plot_ppc_rootograms : Plot ppc rootogram for discrete (count) data
    plot_forest : Plot forest plot for posterior/prior groups.

    Examples
    --------
    Plot posterior predictive intervals for the radon dataset, with custom styling.

    .. plot::
        :context: close-figs

        >>> from arviz_base import load_arviz_data
        >>> import arviz_plots as azp
        >>> azp.style.use("arviz-variat")
        >>> data = load_arviz_data("radon")
        >>> data_subset = data.isel(obs_id=range(50))
        >>> pc = azp.plot_ppc_intervals(
        >>>     data_subset,
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
        raise ValueError("ci_probs must have two elements for twig and trunk intervals.")
    if np.any(ci_probs < 0) or np.any(ci_probs > 1):
        raise ValueError("ci_probs must be between 0 and 1.")
    ci_probs.sort()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    labeller = BaseLabeller()


    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    bg_color = plot_bknd.get_background_color()
    contrast_color = get_contrast_colors(bg_color=bg_color)



    ds_predictive = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )


    if ci_kind == "eti":
        ci_fun = ds_predictive.azstats.eti
    elif ci_kind == "hdi":
        ci_fun = ds_predictive.azstats.hdi

    ci_trunk = ci_fun(prob=ci_probs[1], dim=sample_dims)
    ci_twig = ci_fun(prob=ci_probs[0], dim=sample_dims)

    if point_estimate == "median":
        point = ds_predictive.median(dim=sample_dims)
    elif point_estimate == "mean":
        point = ds_predictive.mean(dim=sample_dims)
    elif point_estimate == "mode":
        point = ds_predictive.azstats.mode(dim=sample_dims)
    else:
        raise ValueError(
            f"point_estimate must be 'mean' or 'median', but {point_estimate} was passed."
        )


    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    bg_color = plot_bknd.get_background_color()
    contrast_color = get_contrast_colors(bg_color=bg_color)

    colors = plot_bknd.get_default_aes("color", 1, {})
    markers = plot_bknd.get_default_aes("marker", 7, {})

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", "__variable__")

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, ds_predictive)

        plot_collection = PlotCollection.wrap(
            ds_predictive,
            backend=backend,
            **pc_kwargs,
        )


    visuals = {} if visuals is None else visuals
    aes_by_visuals = {} if aes_by_visuals is None else aes_by_visuals


    ## trunk intervals
    ci_trunk_kwargs = get_visual_kwargs(visuals, "trunk")
    _, ci_trunk_aes, ci_trunk_ignore = filter_aes(plot_collection, aes_by_visuals, "trunk", sample_dims)

    if ci_trunk_kwargs is not False:
        if "color" not in ci_trunk_aes:
            ci_trunk_kwargs.setdefault("color", colors[0])

        ci_trunk_kwargs.setdefault("alpha", 0.3)
        ci_trunk_kwargs.setdefault("width", 3)

        plot_collection.map(
            ci_bound_y,
            "ci_trunk",
            data=ci_trunk,
            ignore_aes=ci_trunk_ignore,
            **ci_trunk_kwargs,
        )

    ## twig intervals
    ci_twig_kwargs = get_visual_kwargs(visuals, "twig")
    _, ci_twig_aes, ci_twig_ignore = filter_aes(plot_collection, aes_by_visuals, "twig", sample_dims)

    if ci_twig_kwargs is not False:
        if "color" not in ci_twig_aes:
            ci_twig_kwargs.setdefault("color", colors[0])

        ci_twig_kwargs.setdefault("alpha", 0.3)
        ci_twig_kwargs.setdefault("width", 3)

        plot_collection.map(
            ci_bound_y,
            "ci_twig",
            data=ci_twig,
            ignore_aes=ci_twig_ignore,
            **ci_twig_kwargs,
        )


    ## observed_markers
    observed_ms_kwargs = get_visual_kwargs(
        visuals, "observed_markers", False if group == "prior_predictive" else None
    )

    if observed_ms_kwargs is not False:
        _, _, observed_ms_ignore = filter_aes(
            plot_collection, aes_by_visuals, "observed_markers", sample_dims
        )
        observed_ms_kwargs.setdefault("color", contrast_color)
        observed_ms_kwargs.setdefault("marker", markers[6])

        plot_collection.map(
            point_y,
            "observed_markers",
            data=point,
            ignore_aes=observed_ms_ignore,
            **observed_ms_kwargs,
        )



    return plot_collection
