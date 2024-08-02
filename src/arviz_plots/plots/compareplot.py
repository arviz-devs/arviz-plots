"""Compare plot code."""
from importlib import import_module

from arviz_base import rcParams


def plot_compare(cmp_df, color="black", target=None, backend=None):
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
    relative_scale : bool, optiona.
        If True scale the ELPD values relative to the best model.
        Defaults to True???
    figsize : (float, float), optional
        If `None`, size is (6, num of models) inches.
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
    if backend is None:
        backend = rcParams["plot.backend"]

    if backend not in ["bokeh", "matplotlib", "plotly"]:
        raise ValueError(
            f"Invalid backend: '{backend}'. Backend must be 'bokeh', 'matplotlib' or 'plotly'"
        )

    p_be = import_module(f"arviz_plots.backend.{backend}")
    _, target = p_be.create_plotting_grid(1)
    linestyle = p_be.get_default_aes("linestyle", 2, {})[-1]

    # Compute positions of yticks
    yticks_pos = list(range(len(cmp_df), 0, -1))
    yticks_pos_double = [tuple(yticks_pos)] * 2

    # Get scale and adjust it if necessary
    scale = cmp_df["scale"].iloc[0]
    if scale == "negative_log":
        scale = "-log"

    # Compute values for standard error bars
    se_tuple = tuple(cmp_df["elpd_loo"] - cmp_df["se"]), tuple(cmp_df["elpd_loo"] + cmp_df["se"])

    # Plot ELPD point statimes
    p_be.scatter(cmp_df["elpd_loo"], yticks_pos, target, color=color)
    # Plot ELPD standard error bars
    p_be.line(se_tuple, yticks_pos_double, target, color=color)

    # Add reference line for the best model
    # make me nicer
    p_be.line(
        (cmp_df["elpd_loo"].iloc[0], cmp_df["elpd_loo"].iloc[0]),
        (yticks_pos[0], yticks_pos[-1]),
        target,
        color=color,
        linestyle=linestyle,
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
