"""Plot loo pit."""
from arviz_base import convert_to_datatree
from arviz_stats.loo import loo_pit

from arviz_plots.plots.ecdf_plot import plot_ecdf_pit


def plot_loo_pit(
    dt,
    ci_prob=None,
    coverage=False,
    method="simulation",
    n_simulations=1000,
    var_names=None,
    filter_vars=None,  # pylint: disable=unused-argument
    group="posterior_predictive",
    coords=None,  # pylint: disable=unused-argument
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
):
    r"""LOO-PIT Δ-ECDF values with simultaneous confidence envelope.

    For a calibrated model the LOO Probability Integral Transform (PIT) values,
    $p(\tilde{y}_i \le y_i \mid y_{-i})$, should be uniformly distributed.
    Where $y_i$ represents the observed data for index $i$ and $\tilde y_i$ represents
    the posterior predictive sample at index $i$. $y_{-i}$ indicates we have left out the
    $i$-th observation. LOO-PIT values are computed using the PSIS-LOO-CV method described
    in [1]_ and [2]_.

    This plot shows the empirical cumulative distribution function (ECDF) of the LOO-PIT values.
    To make the plot easier to interpret, we plot the Δ-ECDF, that is, the difference between the
    observed ECDF and the expected CDF. Simultaneous confidence bands are computed using the method
    described in described in [3]_.

    Alternatively, we can visualize the coverage of the central posterior credible intervals by
    setting ``coverage=True``. This allows us to assess whether the credible intervals includes
    the observed values. We can obtain the coverage of the central intervals from the LOO-PIT by
    replacing the LOO-PIT with two times the absolute difference between the LOO-PIT values and 0.5.

    For more details on how to interpret this plot,
    see https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#pit-ecdfs.

    Parameters
    ----------
    dt : DataTree
        Input data
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
    var_names : str or list of str, optional
        One or more variables to be plotted. Currently only one variable is supported.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    coords : dict, optional
        Coordinates to plot.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
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

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Plot the ecdf-PIT for the crabs hurdle-negative-binomial dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_loo_pit, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('radon')
        >>> plot_loo_pit(dt)


    Plot the coverage for the crabs hurdle-negative-binomial dataset.

    .. plot::
        :context: close-figs

        >>> plot_loo_pit(dt, coverage=True)


    .. minigallery:: plot_loo_pit

    .. [1] Säilynoja T, Bürkner PC. and Vehtari A. *Graphical test for discrete uniformity and
    its applications in goodness-of-fit evaluation and multiple sample comparison*.
    Statistics and Computing 32(32). (2022) https://doi.org/10.1007/s11222-022-10090-6
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    if group != "posterior_predictive":
        raise ValueError(f"Group {group} not supported. Only 'posterior_predictive' is supported.")

    lpv = loo_pit(dt)
    new_dt = convert_to_datatree(lpv, group="loo_pit")

    plot_kwargs.setdefault("ylabel", {})
    plot_kwargs.setdefault("remove_axis", False)
    plot_kwargs.setdefault("xlabel", {"text": "LOO-PIT"})

    plot_collection = plot_ecdf_pit(
        new_dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group="loo_pit",
        coords=coords,
        sample_dims=lpv.dims,
        ci_prob=ci_prob,
        coverage=coverage,
        n_simulations=n_simulations,
        method=method,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_map=aes_map,
        plot_kwargs=plot_kwargs,
        pc_kwargs=pc_kwargs,
    )

    return plot_collection
