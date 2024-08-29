"""Compare plot code."""
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from datatree import DataTree
from xarray import Dataset

from arviz_plots.plot_collection import PlotCollection


def plot_compare(
    cmp_df, similar_shade=True, relative_scale=False, backend=None, plot_kwargs=None, pc_kwargs=None
):
    r"""Summary plot for model comparison.

    Models are compared based on their expected log pointwise predictive density (ELPD).

    Notes
    -----
    The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
    cross-validation (LOO) or using the widely applicable information criterion (WAIC).
    We recommend LOO in line with the work presented by [1]_.

    Parameters
    ----------
    comp_df : pandas.DataFrame
        Result of the :func:`arviz.compare` method.
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
    plot_kwargs : mapping of {str : mapping or False}, optional
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
    plot_elpd : Plot pointwise elpd differences between two or more models.
    compare : Compare models based on PSIS-LOO loo or WAIC waic cross-validation.
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).
    waic : Compute the widely applicable information criterion.

    References
    ----------
    .. [1] Vehtari et al. (2016). Practical Bayesian model evaluation using leave-one-out
       cross-validation and WAIC https://arxiv.org/abs/1507.04544
    """
    # Check if cmp_df contains the required information criterion
    information_criterion = ["elpd_loo", "elpd_waic"]
    column_index = [c.lower() for c in cmp_df.columns]
    for i_c in information_criterion:
        if i_c in column_index:
            break
    else:
        raise ValueError(
            "cmp_df must contain one of the following "
            f"information criterion: {information_criterion}"
        )

    # Set default backend
    if backend is None:
        backend = rcParams["plot.backend"]

    if plot_kwargs is None:
        plot_kwargs = {}

    if pc_kwargs is None:
        pc_kwargs = {}

    # Get plotting backend
    p_be = import_module(f"arviz_plots.backend.{backend}")

    # Get figure params and create figure and axis
    pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
    figsize = pc_kwargs.get("plot_grid_kws", {}).get("figsize", (2000, len(cmp_df) * 200))
    figsize_units = pc_kwargs.get("plot_grid_kws", {}).get("figsize_units", "dots")
    chart, target = p_be.create_plotting_grid(1, figsize=figsize, figsize_units=figsize_units)

    # Create plot collection
    plot_collection = PlotCollection(
        Dataset({}),
        viz_dt=DataTree.from_dict(
            {
                "/": Dataset(
                    {"chart": np.array(chart, dtype=object), "plot": np.array(target, dtype=object)}
                )
            }
        ),
        backend=backend,
        **pc_kwargs,
    )

    if isinstance(target, np.ndarray):
        target = target.tolist()

    # Set scale relative to the best model
    if relative_scale:
        cmp_df = cmp_df.copy()
        cmp_df[i_c] = cmp_df[i_c] - cmp_df[i_c].iloc[0]

    # Compute positions of yticks
    yticks_pos = list(range(len(cmp_df), 0, -1))

    # Get scale and adjust it if necessary
    scale = cmp_df["scale"].iloc[0]
    if scale == "negative_log":
        scale = "-log"

    # Plot ELPD standard error bars
    if (error_kwargs := plot_kwargs.get("error_bar", {})) is not False:
        error_kwargs.setdefault("color", "black")

        # Compute values for standard error bars
        se_list = list(zip((cmp_df[i_c] - cmp_df["se"]), (cmp_df[i_c] + cmp_df["se"])))

        for se_vals, ytick in zip(se_list, yticks_pos):
            p_be.line(se_vals, (ytick, ytick), target, **error_kwargs)

    # Add reference line for the best model
    if (ref_kwargs := plot_kwargs.get("ref_line", {})) is not False:
        ref_kwargs.setdefault("color", "gray")
        ref_kwargs.setdefault("linestyle", p_be.get_default_aes("linestyle", 2, {})[-1])
        p_be.line(
            (cmp_df[i_c].iloc[0], cmp_df[i_c].iloc[0]),
            (yticks_pos[0], yticks_pos[-1]),
            target,
            **ref_kwargs,
        )

    # Plot ELPD point estimates
    if (pe_kwargs := plot_kwargs.get("point_estimate", {})) is not False:
        pe_kwargs.setdefault("color", "black")
        p_be.scatter(cmp_df[i_c], yticks_pos, target, **pe_kwargs)

    # Add shade for statistically undistinguishable models
    if similar_shade and (shade_kwargs := plot_kwargs.get("shade", {})) is not False:
        shade_kwargs.setdefault("color", "black")
        shade_kwargs.setdefault("alpha", 0.1)

        if scale == "log":
            x_0, x_1 = cmp_df[i_c].iloc[0] - 4, cmp_df[i_c].iloc[0]
        else:
            x_0, x_1 = cmp_df[i_c].iloc[0], cmp_df[i_c].iloc[0] + 4

        padding = (yticks_pos[0] - yticks_pos[-1]) * 0.05
        p_be.fill_between_y(
            x=[x_0, x_1],
            y_bottom=yticks_pos[-1] - padding,
            y_top=yticks_pos[0] + padding,
            target=target,
            **shade_kwargs,
        )

    # Add title and labels
    if (title_kwargs := plot_kwargs.get("title", {})) is not False:
        p_be.title(
            f"Model comparison\n{'higher' if scale == 'log' else 'lower'} is better",
            target,
            **title_kwargs,
        )

    if (labels_kwargs := plot_kwargs.get("labels", {})) is not False:
        p_be.ylabel("ranked models", target, **labels_kwargs)
        p_be.xlabel(f"ELPD ({scale})", target, **labels_kwargs)

    if (ticklabels_kwargs := plot_kwargs.get("ticklabels", {})) is not False:
        p_be.yticks(yticks_pos, cmp_df.index, target, **ticklabels_kwargs)

    return plot_collection
