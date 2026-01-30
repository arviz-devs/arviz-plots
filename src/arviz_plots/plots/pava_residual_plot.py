"""Posterior predictive check for residuals using PAV-adjusted calibration."""
import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.helper_stats import isotonic_fit

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, get_visual_kwargs, set_wrap_layout
from arviz_plots.visuals import (
    fill_between_y,
    labelled_title,
    labelled_x,
    labelled_y,
    line_x,
    scatter_xy,
)


def plot_ppc_pava_residuals(
    dt,
    *,
    x_var,
    var_names=None,
    filter_vars=None,  # pylint: disable=unused-argument
    group="posterior_predictive",
    coords=None,  # pylint: disable=unused-argument
    sample_dims=None,
    data_type="binary",
    ci_prob=None,
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
        Mapping[str, Any] | bool,
    ] = None,
    **pc_kwargs,
):
    """PAV-adjusted calibration residual plot.

    Uses the pool adjacent violators (PAV) algorithm for isotonic regression and computes
    residuals as the difference between the calibrated event probabilities (CEP) and the
    predicted probabilities. A horizontal line at zero corresponds to perfect calibration.
    Details are discussed in [1]_ and [2]_.

    Parameters
    ----------
    dt : DataTree
        Input data
    x_var : array-like, series, DataArray, or str
        Variable to use for x-axis. If a string is given, it should be the name of a variable
        in the `constant_data` group.
    data_type : str
        Defaults to "binary". Other options are "categorical" and "ordinal".
        If "categorical", the plot will show the "one-vs-others" calibration and generate one plot
        per category. If "ordinal", the plot will display cumulative conditional event
        probabilities and generate (number of categories - 1) plots.
    ci_prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``.
    var_names : str or list of str, optional
        One or more variables to be plotted. Currently only one variable is supported.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, "like", "regex"}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
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

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * reference_line -> passed to :func:`~arviz_plots.visuals.line_x`
        * credible_interval -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

        markers defaults to True for residual plots.
        Pass False to disable markers.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Plot the PAVA residual plot for the zeros and non-zeros in a
    negative bimomial model of the roaches dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_pava_residuals, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('roaches_nb')
        >>> plot_ppc_pava_residuals(dt,
        >>>     var_names="y_pos",
        >>>     x_var="roach count")

    .. minigallery:: plot_ppc_pava_residuals

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

    if labeller is None:
        labeller = BaseLabeller()

    visuals.setdefault("markers", True)

    if group == "prior_predictive":
        warnings.warn(
            "\n`plot_ppc_pava_residuals` always use the `observed_data` group."
            "\nBe cautious when using it for prior predictive checks.",
            UserWarning,
            stacklevel=2,
        )

    if isinstance(x_var, str):
        x_val_name = x_var
        x_var = dt.constant_data[x_var].values
    elif hasattr(x_var, "values"):
        x_val_name = x_var.name
        x_var = x_var.values
    else:
        x_val_name = "x"

    ds_residuals = isotonic_fit(
        dt, var_names, group, ci_prob, data_type, residuals=True, x_var=x_var
    )

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", ["__variable__"])
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, ds_residuals)

        plot_collection = PlotCollection.wrap(
            ds_residuals,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    ## reference line at zero
    reference_ls_kwargs = get_visual_kwargs(visuals, "reference_line")

    if reference_ls_kwargs is not False:
        _, _, reference_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "reference_line", sample_dims
        )
        reference_ls_kwargs.setdefault("color", "B2")
        reference_ls_kwargs.setdefault("linestyle", "C1")
        reference_ls_kwargs.setdefault("y", 0)

        plot_collection.map(
            line_x,
            "reference_line",
            data=ds_residuals.sel(plot_axis="x"),
            ignore_aes=reference_ls_ignore,
            **reference_ls_kwargs,
        )

    ## credible interval
    ci_kwargs = get_visual_kwargs(visuals, "credible_interval")
    _, _, ci_ignore = filter_aes(plot_collection, aes_by_visuals, "credible_interval", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", "C0")
        ci_kwargs.setdefault("alpha", 0.25)

        plot_collection.map(
            fill_between_y,
            "credible_interval",
            data=ds_residuals,
            x=ds_residuals.sel(plot_axis="x"),
            y_bottom=ds_residuals.sel(plot_axis="y_bottom"),
            y_top=ds_residuals.sel(plot_axis="y_top"),
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    ## markers
    residual_ms_kwargs = get_visual_kwargs(visuals, "markers")

    if residual_ms_kwargs is not False:
        _, _, residual_ms_ignore = filter_aes(
            plot_collection, aes_by_visuals, "markers", sample_dims
        )
        residual_ms_kwargs.setdefault("color", "C0")
        residual_ms_kwargs.setdefault("marker", "C6")

        plot_collection.map(
            scatter_xy,
            "markers",
            data=ds_residuals,
            ignore_aes=residual_ms_ignore,
            **residual_ms_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "xlabel", sample_dims
    )
    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "B1")

        xlabel_kwargs.setdefault("text", x_val_name)

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

        ylabel_kwargs.setdefault("text", "CEP residual")

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

    return plot_collection
