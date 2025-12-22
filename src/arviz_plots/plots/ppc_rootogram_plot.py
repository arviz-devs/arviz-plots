"""Plot ppc rootogram for discrete (count) data."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.helper_stats import point_interval_unique, point_unique

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    ci_line_y,
    grid,
    labelled_title,
    labelled_x,
    labelled_y,
    scatter_xy,
    set_y_scale,
)


def plot_ppc_rootogram(
    dt,
    ci_prob=None,
    point_estimate=None,
    yscale="sqrt",
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "predictive_markers",
            "observed_markers",
            "credible_interval",
            "xlabel",
            "ylabel",
            "grid",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "predictive_markers",
            "observed_markers",
            "credible_interval",
            "xlabel",
            "ylabel",
            "grid",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    **pc_kwargs,
):
    """Rootogram with confidence intervals per predicted count.

    Rootograms are useful to check the calibration of count models.
    A rootogram shows the difference between observed and predicted counts. The y-axis,
    showing frequencies, is on the square root scale. This makes easier to compare
    observed and expected frequencies even for low frequencies [1]_ and [2]_.

    For more details on how to interpret this plot,
    see https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html

    Parameters
    ----------
    dt : DataTree
        If group is "posterior_predictive", it should contain the ``posterior_predictive`` and
        ``observed_data`` groups. If group is "prior_predictive", it should contain the
        ``prior_predictive`` group.
    ci_prob : float, optional
        Probability for the credible interval. Defaults to rcParam :data:`stats.ci_prob`.
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to rcParam :data:`stats.point_estimate`
    yscale : str, optional
        Scale for the y-axis. Defaults to "sqrt", pass "linear" for linear scale.
        Currently only "matplotlib" backend is supported. For "bokeh" and "plotly"
        the y-axis is linear.
    var_names : str or list of str, optional
        One or more variables to be plotted. Currently only one variable is supported.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str,
        Group to be plotted. Defaults to "posterior_predictive".
        It could also be "prior_predictive".
    coords : dict, optional
        Coordinates to plot.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to rcParam :data:`data.sample_dims`.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * predictive_markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * observed_markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`.
        * credible_interval -> passed to :func:`~arviz_plots.visuals.ci_line_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * grid -> passed to :func:`~arviz_plots.visuals.grid`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

        observed_markers defaults to False, no observed data is plotted, if group is
        "prior_predictive". Pass an (empty) mapping to plot the observed data.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Plot the rootogram for the crabs dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_rootogram, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('crabs_poisson')
        >>> plot_ppc_rootogram(dt)


    .. minigallery:: plot_ppc_rootogram

    References
    ----------
    .. [1] Kleiber C, Zeileis A. *Visualizing Count Data Regressions Using Rootograms*.
        The American Statistician, 70(3). (2016) https://doi.org/10.1080/00031305.2016.1173590

    .. [2] Säilynoja et al. *Recommendations for visual predictive checks in Bayesian workflow*.
        (2025) arXiv preprint https://arxiv.org/abs/2503.01509
    """
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if point_estimate is None:
        point_estimate = rcParams["stats.point_estimate"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if labeller is None:
        labeller = BaseLabeller()

    predictive_dist = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    predictive_types = [
        predictive_dist[var].values.dtype.kind == "f" for var in predictive_dist.data_vars
    ]

    if "observed_data" in dt:
        observed_dist = process_group_variables_coords(
            dt,
            group="observed_data",
            var_names=var_names,
            filter_vars=filter_vars,
            coords=coords,
        )

        observed_types = [
            observed_dist[var].values.dtype.kind == "f" for var in observed_dist.data_vars
        ]
        observed_ds = point_unique(dt, observed_dist.data_vars)
    else:
        observed_types = []

    if any(predictive_types + observed_types):
        raise ValueError(
            "Detected at least one continuous variable.\n"
            "This function only works for discrete (count) data.\n"
            "Consider using other functions such as plot_ppc_dist\n"
            "plot_ppc_pit, or plot_ppc_tstat.",
        )

    predictive_ds = point_interval_unique(
        dt, predictive_dist.data_vars, group, ci_prob, point_estimate
    )

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", "__variable__")

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, predictive_ds)

        plot_collection = PlotCollection.wrap(
            predictive_ds,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    aes_by_visuals.setdefault("predictive_markers", plot_collection.aes_set)
    aes_by_visuals.setdefault("credible_interval", plot_collection.aes_set)
    ## predictive_markers
    predictive_ms_kwargs = get_visual_kwargs(visuals, "predictive_markers")

    if predictive_ms_kwargs is not False:
        _, predictive_ms_aes, predictive_ms_ignore = filter_aes(
            plot_collection, aes_by_visuals, "predictive_markers", sample_dims
        )
        if "color" not in predictive_ms_aes:
            predictive_ms_kwargs.setdefault("color", "C0")

        predictive_ms_kwargs.setdefault("marker", "C4")

        plot_collection.map(
            scatter_xy,
            "predictive_markers",
            data=predictive_ds,
            ignore_aes=predictive_ms_ignore,
            **predictive_ms_kwargs,
        )

    ## confidence intervals
    ci_kwargs = get_visual_kwargs(visuals, "credible_interval")
    _, ci_aes, ci_ignore = filter_aes(
        plot_collection, aes_by_visuals, "credible_interval", sample_dims
    )

    if ci_kwargs is not False:
        if "color" not in ci_aes:
            ci_kwargs.setdefault("color", "C0")

        ci_kwargs.setdefault("alpha", 0.3)
        ci_kwargs.setdefault("width", 3)

        plot_collection.map(
            ci_line_y,
            "credible_interval",
            data=predictive_ds,
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    ## observed_markers
    observed_ms_kwargs = get_visual_kwargs(
        visuals, "observed_markers", False if group == "prior_predictive" else None
    )

    if observed_ms_kwargs is not False:
        _, _, observed_ms_ignore = filter_aes(
            plot_collection, aes_by_visuals, "observed_markers", sample_dims
        )
        observed_ms_kwargs.setdefault("color", "B1")
        observed_ms_kwargs.setdefault("marker", "C6")

        plot_collection.map(
            scatter_xy,
            "observed_markers",
            data=observed_ds,
            ignore_aes=observed_ms_ignore,
            **observed_ms_kwargs,
        )

    ## grid
    grid_kwargs = get_visual_kwargs(visuals, "grid")

    if grid_kwargs is not False:
        _, _, grid_ignore = filter_aes(plot_collection, aes_by_visuals, "grid", sample_dims)
        grid_kwargs.setdefault("color", "B3")
        grid_kwargs.setdefault("axis", "y")

        plot_collection.map(
            grid,
            "grid",
            ignore_aes=grid_ignore,
            **grid_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "xlabel", sample_dims
    )
    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "B1")

        xlabel_kwargs.setdefault("text", "counts")

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=xlabels_ignore,
            subset_info=True,
            **xlabel_kwargs,
        )

    # set ylabel
    _, ylabels_aes, ylabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "ylabel", sample_dims
    )
    ylabel_kwargs = get_visual_kwargs(visuals, "ylabel")
    if ylabel_kwargs is not False:
        if "color" not in ylabels_aes:
            ylabel_kwargs.setdefault("color", "B1")

        ylabel_kwargs.setdefault("text", "frequency")

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=ylabels_ignore,
            subset_info=True,
            **ylabel_kwargs,
        )

    # title
    title_kwargs = get_visual_kwargs(visuals, "title")
    _, _, title_ignore = filter_aes(plot_collection, aes_by_visuals, "title", sample_dims)

    if title_kwargs is not False:
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    plot_collection.map(
        set_y_scale,
        store_artist=backend == "none",
        ignore_aes=plot_collection.aes_set,
        scale=yscale,
    )

    return plot_collection
