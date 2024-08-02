"""Compare plot code."""
from arviz_base import rcParams

from arviz_plots.visuals import (
    labelled_title,
    labelled_x,
    labelled_y,
    line_x,
    line_y,
    scatter_x,
    yticks,
)


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

    # Maybe all this should be in a separate function
    # that also checks what backends are supported and if they can be imported
    if backend == "bokeh":
        from bokeh.plotting import figure

        target = figure()
    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        _, target = plt.subplots()
    elif backend == "plotly":
        import plotly.graph_objects as go

        target = go.Figure()

    # Compute positions of yticks
    yticks_pos = range(len(cmp_df), 0, -1)
    yticks_pos_double = [tuple(yticks_pos)] * len(cmp_df)

    # Get scale and adjust it if necessary
    scale = cmp_df["scale"].iloc[0]
    if scale == "negative_log":
        scale = "-log"

    # Compute values for standard error bars
    se_tuple = tuple(cmp_df["elpd_loo"] - cmp_df["se"]), tuple(cmp_df["elpd_loo"] + cmp_df["se"])

    # Plot ELPD point statimes
    scatter_x(cmp_df["elpd_loo"], target, backend, y=yticks_pos, color=color)
    # Plot ELPD standard error bars
    line_x(se_tuple, target, backend, y=yticks_pos_double, color=color)

    # Add reference line for the best model
    line_y(
        yticks_pos, target, "matplotlib", cmp_df["elpd_loo"].iloc[0], color=color, linestyle="--"
    )

    # Add title and labels
    labelled_title(
        None,
        target,
        backend,
        text=f"Model comparison\n{'higher' if scale == 'log' else 'lower'} is better",
    )
    labelled_y(None, target, backend, text="ranked models")
    labelled_x(None, target, backend, text=f"ELPD ({scale})")
    yticks(None, target, backend, yticks_pos, cmp_df.index)

    return target
