"""Compare plot code."""
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal

import numpy as np
from arviz_base import rcParams
from arviz_base.validate import validate_dict_argument
from xarray import Dataset, DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import get_visual_kwargs


def plot_compare(
    cmp_df,
    *,
    relative_scale=False,
    rotated=False,
    hide_top_model=False,
    backend=None,
    visuals: Mapping[
        Literal[
            "point_estimate",
            "error_bar",
            "ref_line",
            "ref_band",
            "similar_line",
            "labels",
            "title",
            "ticklabels",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    **pc_kwargs,
):
    r"""Summary plot for model comparison.

    Models are compared based on their expected log pointwise predictive density (ELPD).
    Or some transformation of it, such as the mean log predictive density (MLPD)
    or the geometric mean predictive density (GMPD).

    Higher ELPD values indicate better predictive performance.

    The ELPD is estimated by Pareto smoothed importance sampling leave-one-out
    cross-validation (LOO). Details are presented in [1]_ and [2]_.

    The ELPD can only be interpreted in relative terms. But differences in ELPD less than 4
    are considered negligible [3]_.

    Parameters
    ----------
    cmp_df : pandas.DataFrame
        Usually this will be the result of the :func:`arviz_stats.compare` function.
        It is assumed that the DataFrame has at least two columns 
        * When ``relative_scale`` is True: one named `elpd_diff`, `mlpd_diff`, 
          or `gmpd_diff`, the other named `dse`, and the index is the model names.
        * When ``relative_scale`` is False: one named `elpd`, `mlpd`, or `gmpd`, 
          the other named `se`, and the index is the model names.
    relative_scale : bool, optional.
        If True, the `stats`_diff and dse values are used instead of `stats` and se.
        Defaults to True.
    rotated : bool, optional
        If True, the plot is rotated, with models on the y-axis and ELPD on the x-axis.
        Defaults to False.
    hide_top_model : bool, optional
        If True, the top model (first row of `cmp_df`) will not appear as a point with error bars
        or in the axis labels. Its performance can still be accessed by the visuals `ref_line`
        and/or `ref_band`. Defaults to False.
    backend : {"bokeh", "matplotlib", "plotly"}
        Select plotting backend. Defaults to rcParams["plot.backend"].
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * point_estimate -> passed to :func:`~arviz_plots.backend.none.scatter`
        * error_bar -> passed to :func:`~arviz_plots.backend.none.line`
        * ref_line -> passed to :func:`~arviz_plots.backend.none.hline` or
          :func:`~arviz_plots.backend.none.vline` depending on the
          ``rotated`` parameter.
        * ref_band -> passed to :func:`~arviz_plots.backend.none.hspan` or
          :func:`~arviz_plots.backend.none.vspan` depending on the
          ``rotated`` parameter. Defaults to ``False``.
        * similar_line -> passed to :func:`~arviz_plots.backend.none.hline` or
          :func:`~arviz_plots.backend.none.vline` depending on the
          ``rotated`` parameter. Defaults to ``False``.
        * labels -> passed to :func:`~arviz_plots.backend.none.xticks` and
          :func:`~arviz_plots.backend.none.yticks`
        * title -> passed to :func:`~arviz_plots.backend.none.title`
        * ticklabels -> passed to :func:`~arviz_plots.backend.none.yticks`


    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection

    See Also
    --------
    :func:`arviz_stats.compare`: Summary plot for model comparison.
    :func:`arviz_stats.loo` : Compute the ELPD using Pareto smoothed importance sampling
        Leave-one-out cross-validation method.

    References
    ----------
    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017).
        https://doi.org/10.1007/s11222-016-9696-4. arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646

    .. [3] Sivula et al. *Uncertainty in Bayesian Leave-One-Out Cross-Validation Based Model
        Comparison*. (2025). https://doi.org/10.48550/arXiv.2008.10296
    """
    if backend is None:
        backend = rcParams["plot.backend"]
    visuals = validate_dict_argument(visuals, (plot_compare, "visuals"))

    # Check we have the required columns
    required_stats_columns = {"elpd_diff", "mlpd_diff", "gmpd_diff"} if relative_scale else {"elpd", "mlpd", "gmpd"}
    required_se_column = "dse" if relative_scale else "se"
    
    valid_stats = [col for col in required_stats_columns if col in cmp_df.columns]
    if not valid_stats:
        raise ValueError(
            f"When relative_scale is {relative_scale}, "
            f"the DataFrame must contain one of the following columns: {", ".join(required_stats_columns)}."
        )
    
    if required_se_column not in cmp_df.columns:
        raise ValueError(
            f"When relative_scale is {relative_scale}, "
            f"the DataFrame must contain a {required_se_column} column for standard errors."
        )
    
    # Get plotting backend
    p_be = import_module(f"arviz_plots.backend.{backend}")

    # Get figure params and create figure and axis
    figure_kwargs = pc_kwargs.pop("figure_kwargs", {}).copy()
    figsize = figure_kwargs.pop("figsize", None)
    figsize_units = figure_kwargs.pop("figsize_units", None)

    figsize = p_be.scale_fig_size(
        figsize,
        rows=int(len(cmp_df) ** 0.5),
        cols=2,
        figsize_units=figsize_units,
    )
    figsize_units = "dots"

    figure, target = p_be.create_plotting_grid(
        1, figsize=figsize, figsize_units=figsize_units, squeeze=False, **figure_kwargs
    )

    # Create plot collection
    plot_collection = PlotCollection(
        Dataset({}),
        viz_dt=DataTree.from_dict(
            {"/": Dataset({"figure": np.array(figure, dtype=object), "plot": target.squeeze()})}
        ),
        backend=backend,
        **pc_kwargs,
    )

    target = target.item()

    # Obtain stats value and se and setup y axis label
    stats = valid_stats[0]
    if relative_scale:
        se_key = "dse"
        label_score = f"{stats.replace("_diff", "").upper()} (relative)"
    else:
        se_key = "se"
        label_score = f"{stats.upper()}"

    perf_stats = cmp_df[stats].values
    ses = cmp_df[se_key].values

    # Create labels for the models
    label_models = cmp_df.index[hide_top_model:]

    # Compute positions of yticks
    yticks_pos = list(range(len(cmp_df) - hide_top_model, 0, -1))

    # Compute positions of the reference line and band
    pos_ref_line = perf_stats[0]
    pos_ref_band = (perf_stats[0] - ses[0], perf_stats[0] + ses[0])

    # Compute values for standard error bars
    se_list = list(
        zip(
            (perf_stats[hide_top_model:] - ses[hide_top_model:]),
            (perf_stats[hide_top_model:] + ses[hide_top_model:]),
        )
    )

    # Compute positions for mean elpd estimates
    if rotated:
        scatter_x = yticks_pos
        scatter_y = perf_stats[hide_top_model:]
    else:
        scatter_x = perf_stats[hide_top_model:]
        scatter_y = yticks_pos

    # Plot ELPD standard error bars
    error_kwargs = get_visual_kwargs(visuals, "error_bar")
    if error_kwargs is not False:
        error_kwargs.setdefault("color", "B1")

        for se_vals, ytick in zip(se_list, yticks_pos):
            if rotated:
                p_be.line((ytick, ytick), se_vals, target, **error_kwargs)
            else:
                p_be.line(se_vals, (ytick, ytick), target, **error_kwargs)

    # Add reference line for the best model
    ref_l_kwargs = get_visual_kwargs(visuals, "ref_line")
    if ref_l_kwargs is not False:
        ref_l_kwargs.setdefault("color", "B2")
        ref_l_kwargs.setdefault("linestyle", p_be.get_default_aes("linestyle", 2, {})[-1])

        if rotated:
            p_be.hline(pos_ref_line, target, **ref_l_kwargs)
        else:
            p_be.vline(pos_ref_line, target, **ref_l_kwargs)

    # Add reference band for the best model
    ref_b_kwargs = get_visual_kwargs(visuals, "ref_band", False)
    if ref_b_kwargs is not False:
        ref_b_kwargs.setdefault("color", "B2")
        ref_b_kwargs.setdefault("alpha", 0.1)

        if rotated:
            p_be.hspan(*pos_ref_band, target=target, **ref_b_kwargs)
        else:
            p_be.vspan(*pos_ref_band, target=target, **ref_b_kwargs)

    # Plot ELPD point estimates
    pe_kwargs = get_visual_kwargs(visuals, "point_estimate")
    if pe_kwargs is not False:
        pe_kwargs.setdefault("color", "B1")
        p_be.scatter(scatter_x, scatter_y, target, **pe_kwargs)

    # Add line for statistically undistinguishable models
    similar_l_kwargs = get_visual_kwargs(visuals, "similar_line", False)
    if similar_l_kwargs is not False:
        similar_l_kwargs.setdefault("color", "B2")
        similar_l_kwargs.setdefault("linestyle", p_be.get_default_aes("linestyle", 3, {})[-1])

        if rotated:
            p_be.hline(perf_stats[0] - 4, target, **similar_l_kwargs)
        else:
            p_be.vline(perf_stats[0] - 4, target, **similar_l_kwargs)

    # Add title and labels
    title_kwargs = get_visual_kwargs(visuals, "title")
    if title_kwargs is not False:
        p_be.title(
            "Model comparison\nhigher is better",
            target,
            **title_kwargs,
        )

    labels_kwargs = get_visual_kwargs(visuals, "labels")
    if labels_kwargs is not False:
        if rotated:
            p_be.ylabel(label_score, target, **labels_kwargs)
        else:
            p_be.xlabel(label_score, target, **labels_kwargs)

    ticklabels_kwargs = get_visual_kwargs(visuals, "ticklabels")
    if ticklabels_kwargs is not False:
        if rotated:
            p_be.xticks(yticks_pos, label_models, target, **ticklabels_kwargs)
        else:
            p_be.yticks(yticks_pos, label_models, target, **ticklabels_kwargs)

    return plot_collection
