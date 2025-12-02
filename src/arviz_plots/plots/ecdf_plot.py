"""Plot PIT Δ-ECDF."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.ecdf_utils import ecdf_pit

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
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
    ci_prob=0.99,
    coverage=False,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[Literal["ecdf_pit"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    """Plot Δ-ECDF.

    Plots the Δ-ECDF, that is the difference between the observed ECDF and the expected CDF.
    It assumes the values in the DataTree have already been transformed to PIT values,
    as in the case of SBC analysis or values from ``arviz_base.loo_pit``.

    Simultaneous confidence bands are computed using the simulation method described in [1]_.

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
    ci_prob : float
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to 0.99.
    coverage : bool, optional
        If True, plot the coverage of the central posterior credible intervals. Defaults to False.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals` except for "remove_axis"

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * ecdf_lines -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * credible_interval -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats : mapping, optional
        Valid keys are:

        * ecdf_pit -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`. Default is
          ``{"n_simulations": 1000}``.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

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
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()
    visuals.setdefault("remove_axis", True)

    if stats is None:
        stats = {}
    else:
        stats = stats.copy()

    ecdf_pit_kwargs = stats.get("ecdf_pit", {}).copy()
    ecdf_pit_kwargs.setdefault("n_simulations", 1000)

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if labeller is None:
        labeller = BaseLabeller()

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if coverage:
        distribution = distribution / distribution.max()
        distribution = 2 * np.abs(distribution - 0.5)

    dt_ecdf = distribution.azstats.ecdf(dim=sample_dims, pit=True)

    # Compute envelope
    dummy_vals_size = np.prod([len(distribution[dims]) for dims in sample_dims])
    dummy_vals = np.linspace(0, 1, dummy_vals_size)
    x_ci, _, lower_ci, upper_ci = ecdf_pit(dummy_vals, ci_prob, **ecdf_pit_kwargs)
    lower_ci = lower_ci - x_ci
    upper_ci = upper_ci - x_ci

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)
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

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    ## ecdf_line
    ecdf_ls_kwargs = get_visual_kwargs(visuals, "ecdf_lines")

    if ecdf_ls_kwargs is not False:
        _, _, ecdf_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ecdf_lines", sample_dims
        )

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

    ci_kwargs = get_visual_kwargs(visuals, "credible_interval")
    _, _, ci_ignore = filter_aes(plot_collection, aes_by_visuals, "credible_interval", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", "B1")
        ci_kwargs.setdefault("alpha", 0.1)

        plot_collection.map(
            fill_between_y,
            "credible_interval",
            data=dt_ecdf,
            x=x_ci,
            y_bottom=lower_ci,
            y_top=upper_ci,
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "xlabel", sample_dims
    )
    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "B1")

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
    _, ylabels_aes, ylabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "ylabel", sample_dims
    )
    ylabel_kwargs = get_visual_kwargs(visuals, "ylabel", False)
    if ylabel_kwargs is not False:
        if "color" not in ylabels_aes:
            ylabel_kwargs.setdefault("color", "B1")

        ylabel_kwargs.setdefault("text", "Δ ECDF")

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

    if visuals.get("remove_axis", True) is not False:
        plot_collection.map(
            remove_axis,
            store_artist=backend == "none",
            axis="y",
            ignore_aes=plot_collection.aes_set,
        )

    return plot_collection
