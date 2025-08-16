"""Compare plot code."""
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal

import numpy as np
from arviz_base import rcParams
from xarray import Dataset, DataTree

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import get_contrast_colors


def plot_compare(
    cmp_df,
    similar_shade=True,
    relative_scale=False,
    backend=None,
    visuals: Mapping[
        Literal[
            "point_estimate", "error_bar", "ref_line", "shade", "labels", "title", "ticklabels"
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    **pc_kwargs,
):
    r"""Summary plot for model comparison.

    Models are compared based on their expected log pointwise predictive density (ELPD).

    The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
    cross-validation (LOO). Details are presented in [1]_ and [2]_.

    Parameters
    ----------
    comp_df : pandas.DataFrame
        Result of the :func:`arviz_stats.compare` method.
    similar_shade : bool, optional
        If True, a shade is drawn to indicate models with similar
        predictive performance to the best model. Following [3]_,
        models are considered similar if the difference in ELPD is
        less than 4. Defaults to True.
    relative_scale : bool, optional.
        If True scale the ELPD values relative to the best model.
        Defaults to False.
    backend : {"bokeh", "matplotlib", "plotly"}
        Select plotting backend. Defaults to rcParams["plot.backend"].
    figsize : tuple of (float, float), optional
        If `None`, size is (10, num of models) inches.
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * point_estimate -> passed to :func:`~arviz_plots.backend.none.scatter`
        * error_bar -> passed to :func:`~arviz_plots.backend.none.line`
        * ref_line -> passed to :func:`~arviz_plots.backend.none.line`
        * shade -> passed to :func:`~arviz_plots.backend.none.fill_between_y`
        * labels -> passed to :func:`~arviz_plots.backend.none.xticks`
          and :func:`~arviz_plots.backend.none.yticks`
        * title -> passed to :func:`~arviz_plots.backend.none.title`
        * ticklabels -> passed to :func:`~arviz_plots.backend.none.yticks`

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    axes :bokeh figure, matplotlib axes or plotly figure

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
    # Check if cmp_df contains the required information

    column_index = [c.lower() for c in cmp_df.columns]

    if "elpd" not in column_index:
        raise ValueError(
            "cmp_df must have been created using the `compare` function from ArviZ-Stats."
        )

    # Set default backend
    if backend is None:
        backend = rcParams["plot.backend"]

    if visuals is None:
        visuals = {}

    # Get plotting backend
    p_be = import_module(f"arviz_plots.backend.{backend}")
    bg_color = p_be.get_background_color()
    contrast_color, contrast_gray_color = get_contrast_colors(bg_color=bg_color, gray_flag=True)

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
        1, figsize=figsize, figsize_units=figsize_units, **figure_kwargs
    )

    # Create plot collection
    plot_collection = PlotCollection(
        Dataset({}),
        viz_dt=DataTree.from_dict(
            {"/": Dataset({"figure": np.array(figure, dtype=object), "plot": target})}
        ),
        backend=backend,
        **pc_kwargs,
    )

    if isinstance(target, np.ndarray):
        target = target.tolist()

    # Set scale relative to the best model
    if relative_scale:
        cmp_df = cmp_df.copy()
        cmp_df["elpd"] = cmp_df["elpd"] - cmp_df["elpd"].iloc[0]

    # Compute positions of yticks
    yticks_pos = list(range(len(cmp_df), 0, -1))

    # Plot ELPD standard error bars
    if (error_kwargs := visuals.get("error_bar", {})) is not False:
        error_kwargs.setdefault("color", contrast_color)

        # Compute values for standard error bars
        se_list = list(zip((cmp_df["elpd"] - cmp_df["se"]), (cmp_df["elpd"] + cmp_df["se"])))

        for se_vals, ytick in zip(se_list, yticks_pos):
            p_be.line(se_vals, (ytick, ytick), target, **error_kwargs)

    # Add reference line for the best model
    if (ref_kwargs := visuals.get("ref_line", {})) is not False:
        ref_kwargs.setdefault("color", contrast_gray_color)
        ref_kwargs.setdefault("linestyle", p_be.get_default_aes("linestyle", 2, {})[-1])
        p_be.line(
            (cmp_df["elpd"].iloc[0], cmp_df["elpd"].iloc[0]),
            (yticks_pos[0], yticks_pos[-1]),
            target,
            **ref_kwargs,
        )

    # Plot ELPD point estimates
    if (pe_kwargs := visuals.get("point_estimate", {})) is not False:
        pe_kwargs.setdefault("color", contrast_color)
        p_be.scatter(cmp_df["elpd"], yticks_pos, target, **pe_kwargs)

    # Add shade for statistically undistinguishable models
    if similar_shade and (shade_kwargs := visuals.get("shade", {})) is not False:
        shade_kwargs.setdefault("color", contrast_color)
        shade_kwargs.setdefault("alpha", 0.1)

        x_0, x_1 = cmp_df["elpd"].iloc[0] - 4, cmp_df["elpd"].iloc[0]

        padding = (yticks_pos[0] - yticks_pos[-1]) * 0.05
        p_be.fill_between_y(
            x=[x_0, x_1],
            y_bottom=np.repeat(yticks_pos[-1], 2) - padding,
            y_top=np.repeat(yticks_pos[0], 2) + padding,
            target=target,
            **shade_kwargs,
        )

    # Add title and labels
    if (title_kwargs := visuals.get("title", {})) is not False:
        p_be.title(
            "Model comparison\nhigher is better",
            target,
            **title_kwargs,
        )

    if (ticklabels_kwargs := visuals.get("ticklabels", {})) is not False:
        p_be.yticks(yticks_pos, cmp_df.index, target, **ticklabels_kwargs)

    if (labels_kwargs := visuals.get("labels", {})) is not False:
        p_be.ylabel("ranked models", target, **labels_kwargs)
        p_be.xlabel("ELPD", target, **labels_kwargs)

    return plot_collection
