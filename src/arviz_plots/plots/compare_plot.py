"""Compare plot code."""
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from xarray import Dataset, DataTree

from arviz_plots.plot_collection import PlotCollection


def plot_compare(
    cmp_df,
    similar_shade=True,
    relative_scale=False,
    backend=None,
    visuals=None,
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
        predictive performance to the best model. Defaults to True.
    relative_scale : bool, optional.
        If True scale the ELPD values relative to the best model.
        Defaults to False.
    backend : {"bokeh", "matplotlib", "plotly"}
        Select plotting backend. Defaults to rcParams["plot.backend"].
    figsize : (float, float), optional
        If `None`, size is (10, num of models) inches.
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * point_estimate -> passed to :func:`~.backend.scatter`
        * error_bar -> passed to :func:`~.backend.line`
        * ref_line -> passed to :func:`~.backend.line`
        * shade -> passed to :func:`~.backend.fill_between_y`
        * labels -> passed to :func:`~.backend.xticks` and :func:`~.backend.yticks`
        * title -> passed to :func:`~.backend.title`
        * ticklabels -> passed to :func:`~.backend.yticks`

    pc_kwargs : mapping
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

    # Get figure params and create figure and axis
    pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
    figsize = pc_kwargs.get("figure_kwargs", {}).get("figsize", None)
    figsize_units = pc_kwargs["figure_kwargs"].get("figsize_units", "inches")

    figsize = p_be.scale_fig_size(
        figsize,
        rows=int(len(cmp_df) ** 0.5),
        cols=2,
        figsize_units=figsize_units,
    )
    figsize_units = "dots"

    figure, target = p_be.create_plotting_grid(1, figsize=figsize, figsize_units=figsize_units)

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
        error_kwargs.setdefault("color", "black")

        # Compute values for standard error bars
        se_list = list(zip((cmp_df["elpd"] - cmp_df["se"]), (cmp_df["elpd"] + cmp_df["se"])))

        for se_vals, ytick in zip(se_list, yticks_pos):
            p_be.line(se_vals, (ytick, ytick), target, **error_kwargs)

    # Add reference line for the best model
    if (ref_kwargs := visuals.get("ref_line", {})) is not False:
        ref_kwargs.setdefault("color", "gray")
        ref_kwargs.setdefault("linestyle", p_be.get_default_aes("linestyle", 2, {})[-1])
        p_be.line(
            (cmp_df["elpd"].iloc[0], cmp_df["elpd"].iloc[0]),
            (yticks_pos[0], yticks_pos[-1]),
            target,
            **ref_kwargs,
        )

    # Plot ELPD point estimates
    if (pe_kwargs := visuals.get("point_estimate", {})) is not False:
        pe_kwargs.setdefault("color", "black")
        p_be.scatter(cmp_df["elpd"], yticks_pos, target, **pe_kwargs)

    # Add shade for statistically undistinguishable models
    if similar_shade and (shade_kwargs := visuals.get("shade", {})) is not False:
        shade_kwargs.setdefault("color", "black")
        shade_kwargs.setdefault("alpha", 0.1)

        x_0, x_1 = cmp_df["elpd"].iloc[0] - 4, cmp_df["elpd"].iloc[0]

        padding = (yticks_pos[0] - yticks_pos[-1]) * 0.05
        p_be.fill_between_y(
            x=[x_0, x_1],
            y_bottom=yticks_pos[-1] - padding,
            y_top=yticks_pos[0] + padding,
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

    if (labels_kwargs := visuals.get("labels", {})) is not False:
        p_be.ylabel("ranked models", target, **labels_kwargs)
        p_be.xlabel("ELPD", target, **labels_kwargs)

    if (ticklabels_kwargs := visuals.get("ticklabels", {})) is not False:
        p_be.yticks(yticks_pos, cmp_df.index, target, **ticklabels_kwargs)

    return plot_collection
