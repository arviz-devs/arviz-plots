"""Plot ppc pit."""
import warnings
from copy import copy
from importlib import import_module

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.ecdf_utils import difference_ecdf_pit
from numpy import unique

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, set_figure_layout
from arviz_plots.visuals import ecdf_line, fill_between_y, labelled_title, labelled_x, labelled_y


def plot_ppc_pit(
    dt,
    ci_prob=0.95,  # Not sure we need to use rcParams here of if we should just use 0.95
    method="simulation",
    n_simulations=1000,
    loo=False,
    var_names=None,
    data_pairs=None,
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
    """Marginal_p_values with simultaneous confidence envelope.

    For a calibrated model, the posterior predictive p-values should be uniformly distributed.
    This plot shows the empirical cumulative distribution function (ECDF) of the p-values.
    To make the plot easier to interpret, we plot the Δ-ECDF, that is, the difference between
    the expected CDF from the observed ECDF. Simultaneous confidence bands are computed using
    the method described in described in [1]_.

    Parameters
    ----------
    dt : DataTree
        Input data
    ci_prob : float, optional
        Probability for the credible interval. Defaults to 0.95.
    n_simulations : int, optional
        Number of simulations to use to compute simultaneous confidence intervals when using the
        `method="simulation"` ignored if method is "optimized". Defaults to 1000.
    method : str, optional
        Method to compute the confidence intervals. Either "simulation" or "optimized".
        Defaults to "simulation".
    loo : bool, optional
        If True, use the leave-one-out cross-validation samples. Defaults to False.
        Requires the `log_likelihood` group to be present in the DataTree.
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

        * ecdf_lines -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * ci -> passed to :func:`~arviz_plots.visuals.ci_line_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Plot the ecdf-PIT for the crabs hurdle-negative-binomial dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_pit, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('crabs_hurdle_nb')
        >>> plot_ppc_pit(dt)


    .. minigallery:: plot_ppc_pit

    .. [1] Säilynoja T, Bürkner PC. and Vehtari A. *Graphical test for discrete uniformity and
    its applications in goodness-of-fit evaluation and multiple sample comparison*.
    Statistics and Computing 32(32). (2022) https://doi.org/10.1007/s11222-022-10090-6
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
        data_pairs = {var_names: var_names}
    if None in data_pairs.keys():
        data_pairs = dict(zip(dt.posterior_predictive.data_vars, dt.observed_data.data_vars))

    predictive_types = [
        dt.posterior_predictive[var].values.dtype.kind == "i" for var in data_pairs.keys()
    ]
    observed_types = [dt.observed_data[var].values.dtype.kind == "i" for var in data_pairs.values()]

    # For discrete data we need to randomize the PIT values
    randomized = predictive_types + observed_types

    if any(randomized):
        if any(
            set(unique(dt.observed_data[var].values)).issubset({0, 1})
            for var in data_pairs.values()
        ):
            warnings.warn(
                "Observed data is binary. Use plot_ppc_pava instead",
                stacklevel=2,
            )

    # We should default to use loo when available
    if loo:
        pass

    ds_ecdf = difference_ecdf_pit(dt, data_pairs, ci_prob, randomized, method, n_simulations)

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    colors = plot_bknd.get_default_aes("color", 1, {})

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs.setdefault("rows", None)

        pc_kwargs = set_figure_layout(pc_kwargs, plot_bknd, ds_ecdf)

        plot_collection = PlotCollection.wrap(
            ds_ecdf,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    ## ecdf_line
    ecdf_ls_kwargs = copy(plot_kwargs.get("ecdf_lines", {}))

    if ecdf_ls_kwargs is not False:
        _, _, ecdf_ls_ignore = filter_aes(plot_collection, aes_map, "ecdf_lines", sample_dims)
        ecdf_ls_kwargs.setdefault("color", colors[0])

        plot_collection.map(
            ecdf_line,
            "ecdf_lines",
            data=ds_ecdf,
            ignore_aes=ecdf_ls_ignore,
            **ecdf_ls_kwargs,
        )

    ci_kwargs = copy(plot_kwargs.get("ci", {}))
    _, _, ci_ignore = filter_aes(plot_collection, aes_map, "ci", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", colors[0])
        ci_kwargs.setdefault("alpha", 0.2)

        plot_collection.map(
            fill_between_y,
            "ci",
            data=ds_ecdf,
            x=ds_ecdf.sel(plot_axis="x"),
            y_bottom=ds_ecdf.sel(plot_axis="y_bottom"),
            y_top=ds_ecdf.sel(plot_axis="y_top"),
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = copy(plot_kwargs.get("xlabel", {}))
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "black")

        xlabel_kwargs.setdefault("text", "PIT")

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

        ylabel_kwargs.setdefault("text", "Δ ECDF")

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
