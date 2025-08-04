"""Plot ppc using PAV-adjusted calibration plot."""
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.helper_stats import isotonic_fit

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, set_wrap_layout
from arviz_plots.visuals import (
    dline,
    fill_between_y,
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
)


def plot_ppc_pava(
    dt,
    data_type="binary",
    n_bootstaps=1000,
    ci_prob=None,
    var_names=None,
    filter_vars=None,  # pylint: disable=unused-argument
    group="posterior_predictive",
    coords=None,  # pylint: disable=unused-argument
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "lines",
            "markers",
            "reference_line",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "lines",
            "markers",
            "reference_line",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    **pc_kwargs,
):
    """PAV-adjusted calibration plot.

    Uses the pool adjacent violators (PAV) algorithm for isotonic regression.
    An a 45-degree line corresponds to perfect calibration.
    Details are discussed in [1]_ and [2]_.

    Parameters
    ----------
    dt : DataTree
        Input data
    data_type : str
        Defaults to "binary". Other options are "categorical" and "ordinal".
        In case of "categorical" the plot will reflect the "one vs others" calibration.
        And it will generate as many plots as there are categories.
        In case of "ordinal" the plot will reflect the cumulative conditional event
        probabilities. And it will generate a number of plot equal to the number of categories-1.
    n_bootstaps : int, optional
        Number of bootstrap samples to use for estimating the confidence intervals.
        defaults to 1000.
    ci_prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``.
    num_samples : int, optional
        Number of samples to use for the plot. Defaults to 100.
    var_names : str or list of str, optional
        One or more variables to be plotted. Currently only one variable is supported.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, optional
        The group from which to get the unique values. Defaults to "posterior_predictive".
        It could also be "prior_predictive". Notice that this plots always use the "observed_data"
        so use with extra care if you are using "prior_predictive".
    coords : dict, optional
        Coordinates to plot. CURRENTLY NOT IMPLEMENTED
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * lines -> passed to :func:`~arviz_plots.visuals.line_xy`
        * markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * reference_line -> passed to :func:`~arviz_plots.visuals.line_xy`
        * credible_interval -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

        markers defaults to False, no markers are plotted.
        Pass an (empty) mapping to plot markers.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Plot the PAVA calibration plot for the rugby dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_pava, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('rugby')
        >>> plot_ppc_pava(dt, ci_prob=0.90)


    .. minigallery:: plot_ppc_pava

    References
    ----------
    .. [1] Säilynoja et al. *Recommendations for visual predictive checks in Bayesian workflow*.
        (2025) arXiv preprint https://arxiv.org/abs/2503.01509

    .. [2] Dimitriadis et al *Stable reliability diagrams for probabilistic classifiers*.
        PNAS, 118(8) (2021). https://doi.org/10.1073/pnas.2016191118
    """
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
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

    labeller = BaseLabeller()

    visuals.setdefault("markers", False)

    ds_calibration = isotonic_fit(dt, var_names, group, n_bootstaps, ci_prob, data_type)

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    colors = plot_bknd.get_default_aes("color", 1, {})
    markers = plot_bknd.get_default_aes("marker", 7, {})
    lines = plot_bknd.get_default_aes("linestyle", 2, {})

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", ["__variable__"])
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, ds_calibration)

        plot_collection = PlotCollection.wrap(
            ds_calibration,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    ## reference line
    reference_ls_kwargs = copy(visuals.get("reference_line", {}))

    if reference_ls_kwargs is not False:
        _, _, reference_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "reference_line", sample_dims
        )
        reference_ls_kwargs.setdefault("color", "grey")
        reference_ls_kwargs.setdefault("linestyle", lines[1])

        plot_collection.map(
            dline,
            "reference_line",
            data=ds_calibration,
            x=ds_calibration.sel(plot_axis="x"),
            ignore_aes=reference_ls_ignore,
            **reference_ls_kwargs,
        )

    ## markers
    calibration_ms_kwargs = copy(visuals.get("markers", {}))

    if calibration_ms_kwargs is not False:
        _, _, calibration_ms_ignore = filter_aes(
            plot_collection, aes_by_visuals, "markers", sample_dims
        )
        calibration_ms_kwargs.setdefault("color", colors[0])
        calibration_ms_kwargs.setdefault("marker", markers[6])

        plot_collection.map(
            scatter_xy,
            "markers",
            data=ds_calibration,
            ignore_aes=calibration_ms_ignore,
            **calibration_ms_kwargs,
        )

    ## lines
    calibration_ls_kwargs = copy(visuals.get("lines", {}))

    if calibration_ls_kwargs is not False:
        _, _, calibration_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "lines", sample_dims
        )
        calibration_ls_kwargs.setdefault("color", colors[0])

        plot_collection.map(
            line_xy,
            "lines",
            data=ds_calibration,
            ignore_aes=calibration_ls_ignore,
            **calibration_ls_kwargs,
        )

    ci_kwargs = copy(visuals.get("credible_interval", {}))
    _, _, ci_ignore = filter_aes(plot_collection, aes_by_visuals, "credible_interval", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", colors[0])
        ci_kwargs.setdefault("alpha", 0.25)

        plot_collection.map(
            fill_between_y,
            "credible_interval",
            data=ds_calibration,
            x=ds_calibration.sel(plot_axis="x"),
            y_bottom=ds_calibration.sel(plot_axis="y_bottom"),
            y_top=ds_calibration.sel(plot_axis="y_top"),
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "xlabel", sample_dims
    )
    xlabel_kwargs = copy(visuals.get("xlabel", {}))
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "black")

        xlabel_kwargs.setdefault("text", "predicted value")

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
    ylabel_kwargs = copy(visuals.get("ylabel", {}))
    if ylabel_kwargs is not False:
        if "color" not in ylabels_aes:
            ylabel_kwargs.setdefault("color", "black")

        ylabel_kwargs.setdefault("text", "CEP")

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=ylabels_ignore,
            subset_info=True,
            **ylabel_kwargs,
        )

    # title
    title_kwargs = copy(visuals.get("title", {}))
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

    return plot_collection
