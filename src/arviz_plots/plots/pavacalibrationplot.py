"""Plot ppc using PAV-adjusted calibration plot."""
from copy import copy
from importlib import import_module

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.helper_stats import isotonic_fit

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, set_figure_layout
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
    n_bootstaps=1000,
    ci_prob=None,
    data_pairs=None,
    var_names=None,
    filter_vars=None,  # pylint: disable=unused-argument
    coords=None,  # pylint: disable=unused-argument
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
):
    """PAV-adjusted calibration plot.

    Uses the pool adjacent violators (PAV) algorithm for isotonic regression.

    Parameters
    ----------
    dt : DataTree
        Input data
    n_bootstaps : int, optional
        Number of bootstrap samples to use for estimating the confidence intervals.
        defaults to 1000.
    ci_prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``.
    data_pairs : dict, optional
        Dictionary of keys prior/posterior predictive data and values observed data variable names.
        If None, it will assume that the observed data and the predictive data have
        the same variable name.
    num_samples : int, optional
        Number of samples to use for the plot. Defaults to 100.
    var_names : str or list of str, optional
        One or more variables to be plotted. Currently only one variable is supported.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    coords : dict, optional
        Coordinates to plot. CURRENTLY NOT IMPLEMENTED
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * lines -> passed to :func:`~arviz_plots.visuals.line_xy`
        * markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * reference_line -> passed to :func:`~arviz_plots.visuals.line_xy`
        * ci -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

        markers defaults to False, no markers are plotted.
        Pass an (empty) mapping to plot markers.

    pc_kwargs : mapping
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
    .. [1] Dimitriadis et al *Stable reliability diagrams for probabilistic classifiers*.
        PNAS, 118(8) (2021). https://doi.org/10.1073/pnas.2016191118

    """
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    labeller = BaseLabeller()

    plot_kwargs.setdefault("markers", False)

    if data_pairs is None:
        data_pairs = {var_names: var_names}

    ds_calibration = isotonic_fit(dt, data_pairs, n_bootstaps, ci_prob)

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    colors = plot_bknd.get_default_aes("color", 1, {})
    markers = plot_bknd.get_default_aes("marker", 7, {})
    lines = plot_bknd.get_default_aes("linestyle", 2, {})

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("rows", None)
        pc_kwargs.setdefault("cols", ["__variable__"])
        pc_kwargs = set_figure_layout(pc_kwargs, plot_bknd, ds_calibration)

        plot_collection = PlotCollection.wrap(
            ds_calibration,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    ## reference line
    reference_ls_kwargs = copy(plot_kwargs.get("reference_line", {}))

    if reference_ls_kwargs is not False:
        _, _, reference_ls_ignore = filter_aes(
            plot_collection, aes_map, "reference_line", sample_dims
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
    calibration_ms_kwargs = copy(plot_kwargs.get("markers", {}))

    if calibration_ms_kwargs is not False:
        _, _, calibration_ms_ignore = filter_aes(plot_collection, aes_map, "markers", sample_dims)
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
    calibration_ls_kwargs = copy(plot_kwargs.get("lines", {}))

    if calibration_ls_kwargs is not False:
        _, _, calibration_ls_ignore = filter_aes(plot_collection, aes_map, "lines", sample_dims)
        calibration_ls_kwargs.setdefault("color", colors[0])

        plot_collection.map(
            line_xy,
            "lines",
            data=ds_calibration,
            ignore_aes=calibration_ls_ignore,
            **calibration_ls_kwargs,
        )

    ci_kwargs = copy(plot_kwargs.get("ci", {}))
    _, _, ci_ignore = filter_aes(plot_collection, aes_map, "ci", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", colors[0])
        ci_kwargs.setdefault("alpha", 0.25)

        plot_collection.map(
            fill_between_y,
            "ci",
            data=ds_calibration,
            x=ds_calibration.sel(plot_axis="x"),
            y_bottom=ds_calibration.sel(plot_axis="y_bottom"),
            y_top=ds_calibration.sel(plot_axis="y_top"),
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = copy(plot_kwargs.get("xlabel", {}))
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
    _, ylabels_aes, ylabels_ignore = filter_aes(plot_collection, aes_map, "ylabel", sample_dims)
    ylabel_kwargs = copy(plot_kwargs.get("ylabel", {}))
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
    title_kwargs = copy(plot_kwargs.get("title", {}))
    _, _, title_ignore = filter_aes(plot_collection, aes_map, "title", sample_dims)

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
