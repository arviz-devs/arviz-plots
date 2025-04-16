"""Plot ppc rootogram for discrete (count) data."""
from copy import copy
from importlib import import_module

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.helper_stats import point_interval_unique, point_unique

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords, set_wrap_layout
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
    yscale="sqrt",
    data_pairs=None,
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
):
    """Rootogram with confidence intervals per predicted count.

    Rootograms are useful to check the calibration of count models.
    A rootogram shows the difference between observed and predicted counts. The y-axis,
    showing frequencies, is on the square root scale. This makes easier to compare
    observed and expected frequencies even for low frequencies [1]_.


    Parameters
    ----------
    dt : DataTree
        Input data
    ci_prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``.
    yscale : str, optional
        Scale for the y-axis. Defaults to "sqrt", pass "linear" for linear scale.
        Currently only "matplotlib" backend is supported. For "bokeh" and "plotly"
        the y-axis is linear.
    data_pairs : dict, optional
        Dictionary of keys prior/posterior predictive data and values observed data variable names.
        If None, it will assume that the observed data and the predictive data have
        the same variable name.
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
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * predictive_markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * observed_markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`. Defaults to
            False if group is "prior_predictive" and {} otherwise.
        * ci -> passed to :func:`~arviz_plots.visuals.ci_line_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * grid -> passed to :func:`~arviz_plots.visuals.grid`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.grid`

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

    predictive_types = [
        predictive_dist[var].values.dtype.kind == "f" for var in predictive_dist.data_vars
    ]
    observed_types = [
        observed_dist[var].values.dtype.kind == "f" for var in observed_dist.data_vars
    ]

    if any(predictive_types + observed_types):
        raise ValueError(
            "Detected at least one continuous variable.\n"
            "Use plot_ppc variants specific for continuous data, "
            "such as plot_ppc_dist.",
        )

    ds_predictive = point_interval_unique(dt, predictive_dist.data_vars, group, ci_prob)
    observed_ds = point_unique(dt, observed_dist.data_vars)

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    colors = plot_bknd.get_default_aes("color", 1, {})
    markers = plot_bknd.get_default_aes("marker", 7, {})

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs.setdefault("rows", None)

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, ds_predictive)

        plot_collection = PlotCollection.wrap(
            ds_predictive,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    aes_map.setdefault("predictive_markers", plot_collection.aes_set)
    aes_map.setdefault("ci", plot_collection.aes_set)
    ## predictive_markers
    predictive_ms_kwargs = copy(plot_kwargs.get("predictive_markers", {}))

    if predictive_ms_kwargs is not False:
        _, predictive_ms_aes, predictive_ms_ignore = filter_aes(
            plot_collection, aes_map, "predictive_markers", sample_dims
        )
        if "color" not in predictive_ms_aes:
            predictive_ms_kwargs.setdefault("color", colors[0])

        predictive_ms_kwargs.setdefault("marker", markers[4])

        plot_collection.map(
            scatter_xy,
            "predictive_markers",
            data=ds_predictive,
            ignore_aes=predictive_ms_ignore,
            **predictive_ms_kwargs,
        )

    ## confidence intervals
    ci_kwargs = copy(plot_kwargs.get("ci", {}))
    _, ci_aes, ci_ignore = filter_aes(plot_collection, aes_map, "ci", sample_dims)

    if ci_kwargs is not False:
        if "color" not in ci_aes:
            ci_kwargs.setdefault("color", colors[0])

        ci_kwargs.setdefault("alpha", 0.3)
        ci_kwargs.setdefault("width", 3)

        plot_collection.map(
            ci_line_y,
            "ci",
            data=ds_predictive,
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    ## observed_markers
    observed_ms_kwargs = copy(
        plot_kwargs.get("observed_markers", False if group == "prior_predictive" else {})
    )

    if observed_ms_kwargs is not False:
        _, _, observed_ms_ignore = filter_aes(
            plot_collection, aes_map, "observed_markers", sample_dims
        )
        observed_ms_kwargs.setdefault("color", "black")
        observed_ms_kwargs.setdefault("marker", markers[6])

        plot_collection.map(
            scatter_xy,
            "observed_markers",
            data=observed_ds,
            ignore_aes=observed_ms_ignore,
            **observed_ms_kwargs,
        )

    ## grid
    grid_kwargs = copy(plot_kwargs.get("grid", {}))

    if grid_kwargs is not False:
        _, _, grid_ignore = filter_aes(plot_collection, aes_map, "grid", sample_dims)
        grid_kwargs.setdefault("color", "#cccccc")
        grid_kwargs.setdefault("axis", "y")

        plot_collection.map(
            grid,
            "grid",
            ignore_aes=grid_ignore,
            **grid_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = copy(plot_kwargs.get("xlabel", {}))
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "black")

        xlabel_kwargs.setdefault("text", "counts")

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

        ylabel_kwargs.setdefault("text", "frequency")

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

    plot_collection.map(
        set_y_scale,
        store_artist=backend == "none",
        ignore_aes=plot_collection.aes_set,
        scale=yscale,
    )

    return plot_collection
