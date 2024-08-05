"""Compare plot code."""
from importlib import import_module

from arviz_base import rcParams


def plot_compare(
    cmp_df,
    color="black",
    similar_band=True,
    relative_scale=False,
    figsize=None,
    target=None,
    backend=None,
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
    color : str, optional
        Color for the plot elements. Defaults to "black".
    similar_band : bool, optional
        If True, a band is drawn to indicate models with similar
        predictive performance to the best model. Defaults to True.
    relative_scale : bool, optional.
        If True scale the ELPD values relative to the best model.
        Defaults to True???
    figsize : (float, float), optional
        If `None`, size is (10, num of models) inches.
    target : bokeh figure, matplotlib axes, or plotly figure optional
    backend : {"bokeh", "matplotlib", "plotly"}
        Select plotting backend. Defaults to rcParams["plot.backend"].

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

    if backend is None:
        backend = rcParams["plot.backend"]

    if backend not in ["bokeh", "matplotlib", "plotly"]:
        raise ValueError(
            f"Invalid backend: '{backend}'. Backend must be 'bokeh', 'matplotlib' or 'plotly'"
        )

    if relative_scale:
        cmp_df = cmp_df.copy()
        cmp_df[i_c] = cmp_df[i_c] - cmp_df[i_c].iloc[0]

    if figsize is None:
        figsize = (10, len(cmp_df))

    p_be = import_module(f"arviz_plots.backend.{backend}")
    _, target = p_be.create_plotting_grid(1, figsize=figsize)
    linestyle = p_be.get_default_aes("linestyle", 2, {})[-1]

    # Compute positions of yticks
    yticks_pos = list(range(len(cmp_df), 0, -1))

    # Get scale and adjust it if necessary
    scale = cmp_df["scale"].iloc[0]
    if scale == "negative_log":
        scale = "-log"

    # Compute values for standard error bars
    # se_tuple = tuple(cmp_df[i_c] - cmp_df["se"]), tuple(cmp_df[i_c] + cmp_df["se"])
    se_list = list(zip((cmp_df[i_c] - cmp_df["se"]), (cmp_df[i_c] + cmp_df["se"])))

    # Plot ELPD point statimes
    p_be.scatter(cmp_df[i_c], yticks_pos, target, color=color)
    # Plot ELPD standard error bars
    for se_vals, ytick in zip(se_list, yticks_pos):
        p_be.line(se_vals, (ytick, ytick), target, color=color)

    # Add reference line for the best model
    p_be.line(
        (cmp_df[i_c].iloc[0], cmp_df[i_c].iloc[0]),
        (yticks_pos[0], yticks_pos[-1]),
        target,
        color=color,
        linestyle=linestyle,
        alpha=0.5,
    )

    # Add band for statistically undistinguishable models
    if similar_band:
        if scale == "log":
            x_0, x_1 = cmp_df[i_c].iloc[0] - 4, cmp_df[i_c].iloc[0]
        else:
            x_0, x_1 = cmp_df[i_c].iloc[0], cmp_df[i_c].iloc[0] + 4

        p_be.fill_between_y(
            x=[x_0, x_1],
            y_bottom=yticks_pos[-1],
            y_top=yticks_pos[0],
            target=target,
            color=color,
            alpha=0.1,
        )

    # Add title and labels
    p_be.title(
        f"Model comparison\n{'higher' if scale == 'log' else 'lower'} is better",
        target,
    )
    p_be.ylabel("ranked models", target)
    p_be.xlabel(f"ELPD ({scale})", target)
    p_be.yticks(yticks_pos, cmp_df.index, target)

    return target
