"""Plot PIT Δ-ECDF."""
from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.ecdf_utils import ecdf_pit

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords, set_wrap_layout
from arviz_plots.visuals import (
    ecdf_line,
    fill_between_y,
    labelled_title,
    labelled_x,
    labelled_y,
    remove_axis,
    set_xticks,
)


def plot_ecdf_pit(
    dt,
    var_names=None,
    filter_vars=None,
    group="prior_sbc",
    coords=None,
    sample_dims=None,
    ci_prob=None,
    coverage=False,
    n_simulations=1000,
    method="simulation",
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
):
    """Plot Δ-ECDF.

    Plots the Δ-ECDF, that is the difference between the observed ECDF and the expected CDF.
    It assumes the values in the DataTree have already been transformed to PIT values,
    as in the case of SBC analysis or values from ``arviz_base.loo_pit``.

    Simultaneous confidence bands are computed using the method described in [1]_.

    Alternatively, we can visualize the coverage of the central posterior credible intervals by
    setting ``coverage=True``. This allows us to assess whether the credible intervals includes
    the observed values. We can obtain the coverage of the central intervals from the PIT by
    replacing the PIT with two times the absolute difference between the PIT values and 0.5.

    For more details on how to interpret this plot,
    see https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#pit-ecdfs.

    Parameters
    ----------
    dt : DataTree
        Input data
    var_names : str or list of str, optional
        One or more variables to be plotted. Currently only one variable is supported.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, optional
        Which group to use. Defaults to "prior_sbc".
    coords : dict, optional
        Coordinates to plot.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to ``rcParams["stats.ci_prob"]``
    coverage : bool, optional
        If True, plot the coverage of the central posterior credible intervals. Defaults to False.
    n_simulations : int, optional
        Number of simulations to use to compute simultaneous confidence intervals when using the
        `method="simulation"` ignored if method is "optimized". Defaults to 1000.
    method : str, optional
        Method to compute the confidence intervals. Either "simulation" or "optimized".
        Defaults to "simulation".
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
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Rank plot for the crabs hurdle-negative-binomial dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ecdf_pit, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('sbc')
        >>> plot_ecdf_pit(dt)


    .. minigallery:: plot_ecdf_pit

    References
    ----------
    .. [1] Säilynoja et al. *Graphical test for discrete uniformity and
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
    plot_kwargs.setdefault("remove_axis", True)
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

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if coverage:
        distribution = distribution / distribution.max()
        distribution = 2 * np.abs(distribution - 0.5)

    dt_ecdf = distribution.azstats.ecdf(dims=sample_dims, pit=True)

    # Compute envelope
    dummy_vals_size = np.prod([len(distribution[dims]) for dims in sample_dims])
    dummy_vals = np.linspace(0, 1, dummy_vals_size)
    x_ci, _, lower_ci, upper_ci = ecdf_pit(dummy_vals, ci_prob, method, n_simulations)
    lower_ci = lower_ci - x_ci
    upper_ci = upper_ci - x_ci

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["plot_grid_kws"].setdefault("sharex", True)
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("col_wrap", 4)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in distribution.dims if dim not in sample_dims]
        )

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, distribution)

        plot_collection = PlotCollection.wrap(
            distribution,
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

        plot_collection.map(
            ecdf_line,
            "ecdf_lines",
            data=dt_ecdf,
            ignore_aes=ecdf_ls_ignore,
            **ecdf_ls_kwargs,
        )

    if coverage:
        plot_collection.map(
            set_xticks,
            "ecdf_xticks",
            values=[0, 0.25, 0.5, 0.75, 1],
            labels=["0", "25", "50", "75", "100"],
            store_artist=backend == "none",
        )

    ci_kwargs = copy(plot_kwargs.get("ci", {}))
    _, _, ci_ignore = filter_aes(plot_collection, aes_map, "ci", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", "black")
        ci_kwargs.setdefault("alpha", 0.1)

        plot_collection.map(
            fill_between_y,
            "ci",
            data=dt_ecdf,
            x=x_ci,
            y_bottom=lower_ci,
            y_top=upper_ci,
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = copy(plot_kwargs.get("xlabel", {}))
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "black")

        if coverage:
            xlabel_kwargs.setdefault("text", "ETI %")
        else:
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
    ylabel_kwargs = copy(plot_kwargs.get("ylabel", False))
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

    if plot_kwargs.get("remove_axis", True) is not False:
        plot_collection.map(
            remove_axis,
            store_artist=backend == "none",
            axis="y",
            ignore_aes=plot_collection.aes_set,
        )

    return plot_collection
