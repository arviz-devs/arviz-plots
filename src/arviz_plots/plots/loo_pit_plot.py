"""Plot loo pit."""
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import xarray as xr
from arviz_base import convert_to_datatree
from arviz_base.validate import validate_dict_argument
from arviz_stats.loo import loo_pit

from arviz_plots.plots.ecdf_plot import plot_ecdf_pit


def plot_loo_pit(
    dt,
    *,
    var_names=None,
    filter_vars=None,  # pylint: disable=unused-argument
    group="posterior_predictive",
    coords=None,  # pylint: disable=unused-argument
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
    r"""LOO-PIT Δ-ECDF uniformity test.

    For a calibrated model the LOO Probability Integral Transform (PIT) values,
    $p(\tilde{y}_i \le y_i \mid y_{-i})$, should be uniformly distributed.
    Where $y_i$ represents the observed data for index $i$ and $\tilde y_i$ represents
    the posterior predictive sample at index $i$. $y_{-i}$ indicates we have left out the
    $i$-th observation. LOO-PIT values are computed using the PSIS-LOO-CV method described
    in [1]_ and [2]_.

    This plot shows the empirical cumulative distribution function (ECDF) of the LOO-PIT values.
    To make the plot easier to interpret, we plot the Δ-ECDF, that is, the difference between the
    observed ECDF and the expected CDF.
    The points that contribute the most to deviations from uniformity are
    computed as described in [3]_ and highlighted in the plot.

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
    envelope_prob : float, optional
        Indicates the probability that should be contained within the envelope.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    coverage : bool, optional
        If True, plot the coverage of the central posterior credible intervals. Defaults to False.
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
        CURRENTLY NOT SUPPORTED
    method : {"pot_c", "prit_c", "piet_c"}, optional
        Method to use for the uniformity test. Defaults to "pot_c".
        Check the documentation of :func:`~arviz_plots.plot_ecdf_pit` for
        more details.
    envelope_prob : float, optional
        Indicates the probability threshold to highlight points.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals` except for "remove_axis".

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * ecdf_lines -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * credible_interval -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis -> not passed anywhere, can only be a boolean to indicate
          whether to call this function. Defaults to ``False`` for plot_loo_pit

    stats : mapping, optional
        Valid keys are:

        * ecdf_pit -> passed to :func:`~xarray.Dataset.azstats.uniformity_test`. Default is
          ``{"gamma": 0}``.


    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    See Also
    --------
    plot_ppc_pit : Predictive check using PIT Δ-ECDF uniformity test.
    plot_loo_interval : Predictive intervals and observed data.

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

    References
    ----------
    .. [1] Vehtari et al. Practical Bayesian model evaluation using leave-one-out cross-validation
       and WAIC. Statistics and Computing. 27(5) (2017) https://doi.org/10.1007/s11222-016-9696-4

    .. [2] Vehtari et al. Pareto Smoothed Importance Sampling. Journal of Machine Learning
       Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html

    .. [3] Tasso et al. *LOO-PIT predictive model checking* arXiv:2603.02928 (2026).
    """
    if group != "posterior_predictive":
        raise ValueError(f"Group {group} not supported. Only 'posterior_predictive' is supported.")
    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_loo_pit, "aes_by_visuals"))
    visuals = validate_dict_argument(visuals, (plot_loo_pit, "visuals"))
    stats = validate_dict_argument(stats, (plot_loo_pit, "stats"))
    if sample_dims is None:
        sample_dims = ["chain", "draw"]
    else:
        warnings.warn(
            "'sample_dims' is currently not supported in plot_loo_pit and will be ignored"
        )

    if method == "envelope":
        warnings.warn(
            "Method 'envelope' is not recommended for the LOO-PIT plot."
            "As it assumes PIT values are independent, which is not the case for LOO-PIT values.",
            UserWarning,
        )

    if method not in {"envelope", "pot_c", "prit_c", "piet_c"}:
        raise ValueError(
            f"Method {method} not supported. Choose from 'envelope', 'pot_c', 'prit_c' or 'piet_c'."
        )

    pareto_pit = method in ["pot_c", "piet_c"]
    lpv = loo_pit(dt, pareto_pit=pareto_pit)
    new_dt = convert_to_datatree(lpv, group="loo_pit")

    visuals.setdefault("ylabel", {})
    visuals.setdefault("remove_axis", False)
    visuals.setdefault("xlabel", {"text": "ETI %" if coverage else "LOO-PIT"})

    plot_collection = plot_ecdf_pit(
        new_dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group="loo_pit",
        coords=coords,
        sample_dims=lpv.dims,
        method=method,
        envelope_prob=envelope_prob,
        coverage=coverage,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats,
        **pc_kwargs,
    )

    return plot_collection
