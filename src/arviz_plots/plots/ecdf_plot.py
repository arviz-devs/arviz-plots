"""Plot PIT Δ-ECDF."""
import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from arviz_stats.ecdf_utils import ecdf_pit

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    annotate_xy,
    ecdf_line,
    fill_between_y,
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    remove_axis,
    scatter_xy,
    set_xticks,
    set_ylim,
)


def plot_ecdf_pit(
    dt,
    *,
    var_names=None,
    filter_vars=None,
    group="prior_sbc",
    coords=None,
    sample_dims=None,
    method="pot_c",
    envelope_prob=None,
    coverage=False,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "suspicious_points",
            "p_value_text",
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
            "suspicious_points",
            "p_value_text",
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
    method : {"pot_c", "prit_c", "piet_c", "envelope"}, optional
        Method to use for the uniformity test. See the "Notes" section for the full description of
        the different methods available.
    envelope_prob : float, optional
        If `method` is "envelope", indicates the probability that should be contained within the
        envelope, otherwise indicates the probability threshold to highlight points.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    coverage : bool, optional
        Defaults to ``rcParams["stats.envelope_prob"]``.
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
        * credible_interval -> passed to :func:`~arviz_plots.visuals.fill_between_y`,
          only when method is "envelope"
        * ref_line -> passed to :func:`~arviz_plots.visuals.line_xy`
        * suspicious_points -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * p_value_text -> passed to :func:`~arviz_plots.visuals.annotate_xy`
          only when method is not "envelope"
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats : mapping, optional
        Valid keys are:

        * ecdf_pit -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`. or
          :func:`~xarray.Dataset.azstats.uniformity_test` depending on the value of `method`.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    Notes
    -----
    The following methods are available for testing the uniformity of the PIT values:

    * pot_c: Good default choice due to its good power against diverse
      type of local departures from the null. Preferred in almost all cases.
    * piet_c: Use when you specifically want to evaluate tail deviations.
    * prit_c: Mostly compatible with PITs computed as normalized ranks.
      Don't use unless you have a specific reason to do so.
    * envelope: Legacy method that uses simultaneous confidence bands. It can be used
      when you have independent PIT values, as in the case of SBC analysis. The method
      is described in method described in [1]_. Notice that pot_c is also valid in those cases.

    The methods "pot_c", "piet_c" and "prit_c" compute the points that contribute the most
    to deviations from uniformity as described in [2]_.


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

    .. [2] Tasso et al. *LOO-PIT predictive model checking* arXiv:2603.02928 (2026).
    """
    envelope_prob = validate_or_use_rcparam(envelope_prob, "stats.envelope_prob")
    alpha = 1 - envelope_prob
    visuals = validate_dict_argument(visuals, (plot_ecdf_pit, "visuals"))
    if method == "envelope":
        visuals.setdefault("remove_axis", True)
    else:
        visuals.setdefault("remove_axis", False)
    stats = validate_dict_argument(stats, (plot_ecdf_pit, "stats"))

    ecdf_pit_kwargs = stats.get("ecdf_pit", {}).copy()
    if method == "envelope":
        ecdf_pit_kwargs.setdefault("n_simulations", 1000)
    else:
        ecdf_pit_kwargs.setdefault("gamma", 0)

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
    sample_dims = validate_sample_dims(sample_dims, data=distribution)
    sample_size = np.prod([distribution.sizes[dim] for dim in sample_dims])

    if method not in {"envelope", "pot_c", "prit_c", "piet_c"}:
        raise ValueError(
            f"Method {method} not supported. Choose from 'envelope', 'pot_c', 'prit_c' or 'piet_c'."
        )
    if method == "envelope":
        warnings.warn(
            "Method 'envelope' will be deprecated. As it assumes PIT values are independent.\n"
            "Use 'pot_c' instead, which is valid for both independent and dependent PIT values.",
            FutureWarning,
        )

    # ensure we have PIT values between 0 and 1.
    dist_max = distribution.max()
    if any(float(dist_max[v]) > 1 for v in distribution.data_vars):
        distribution = (distribution + 0.5) / (dist_max + 1)

    if coverage:
        distribution = 2 * np.abs(distribution - 0.5)

    dt_ecdf = distribution.azstats.ecdf(dim=sample_dims, pit=True, npoints=sample_size)
    x_ci = lower_ci = upper_ci = None

    if method == "envelope":
        dummy_vals = np.linspace(0, 1, sample_size)
        x_ci, _, lower_ci, upper_ci = ecdf_pit(dummy_vals, envelope_prob, **ecdf_pit_kwargs)
        lower_ci = lower_ci - x_ci
        upper_ci = upper_ci - x_ci

    else:
        p_values, shapley_vals = distribution.azstats.uniformity_test(
            dim=sample_dims, method=method
        )

        gamma = stats.get("ecdf_pit", {}).get("gamma", 0)
        highlight = (shapley_vals > gamma) & (p_values < alpha)
        suspicious_mask = highlight.rename({"pit_dim": "ecdf_dim"})
        # use the Dvoretzky-Kiefer-Wolfowitz inequality plus a small padding
        # to get the default y-limits for the plot.
        expected_max = np.sqrt(np.log(2 / alpha) / (2 * sample_size)) * 1.3

        actual_max = np.max(np.abs(dt_ecdf.sel(plot_axis="y").to_array())).item()
        epsilon = max(expected_max, actual_max)

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)
        pc_kwargs["figure_kwargs"].setdefault("sharey", True)
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

    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_ecdf_pit, "aes_by_visuals"))

    ## reference line
    ref_ls_kwargs = get_visual_kwargs(visuals, "ref_line")
    if ref_ls_kwargs is not False:
        _, _, ref_ls_ignore = filter_aes(plot_collection, aes_by_visuals, "ref_line", sample_dims)
        ref_ls_kwargs.setdefault("color", "B3")
        ref_ls_kwargs.setdefault("linestyle", "C1")

        plot_collection.map(
            line_xy,
            "ref_line",
            data=dt_ecdf.sel(plot_axis="x"),
            x=[0, 1],
            y=0,
            ignore_aes=ref_ls_ignore,
            **ref_ls_kwargs,
        )

    ## ecdf_line
    ecdf_ls_kwargs = get_visual_kwargs(visuals, "ecdf_lines")

    if ecdf_ls_kwargs is not False:
        _, ecdf_ls_aes, ecdf_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ecdf_lines", sample_dims
        )
        if "color" not in ecdf_ls_aes:
            ecdf_ls_kwargs.setdefault("color", "C0")

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

    if method == "envelope":
        ci_kwargs = get_visual_kwargs(visuals, "credible_interval")
        _, _, ci_ignore = filter_aes(
            plot_collection, aes_by_visuals, "credible_interval", sample_dims
        )
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
                step=True,
                ignore_aes=ci_ignore,
                **ci_kwargs,
            )
    else:
        suspicious_kwargs = get_visual_kwargs(visuals, "suspicious_points")
        _, suspicious_aes, suspicious_ignore = filter_aes(
            plot_collection, aes_by_visuals, "suspicious_points", sample_dims
        )
        if suspicious_kwargs is not False:
            if "color" not in suspicious_aes:
                suspicious_kwargs.setdefault("color", "C1")
            if "marker" not in suspicious_aes:
                suspicious_kwargs.setdefault("marker", "C6")

            plot_collection.map(
                scatter_xy,
                "suspicious_points",
                data=dt_ecdf,
                mask=suspicious_mask,
                ignore_aes=suspicious_ignore,
                **suspicious_kwargs,
            )
        plot_collection.map(
            set_ylim,
            "ylim",
            limits=(-epsilon, epsilon),
            store_artist=False,
            ignore_aes="all",
        )
        # add p-values as annotations
        p_value_kwargs = get_visual_kwargs(visuals, "p_value_text")
        if p_value_kwargs is not False:
            _, _, p_value_ignore = filter_aes(
                plot_collection, aes_by_visuals, "p_value_text", sample_dims
            )
            p_value_kwargs.setdefault("text", lambda p: f"p={p:.2f}(α={alpha:.2f}) ")
            p_value_kwargs.setdefault("x", 0)
            p_value_kwargs.setdefault("y", 0.85 * epsilon)
            p_value_kwargs.setdefault("horizontal_align", "left")

            plot_collection.map(
                annotate_xy,
                "p_value_text",
                data=p_values,
                ignore_aes=p_value_ignore,
                store_artist=backend == "none",
                **p_value_kwargs,
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
