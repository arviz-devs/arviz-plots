"""Plot ppc pit."""
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from arviz_stats.base.array import array_stats

from arviz_plots.plots import plot_ecdf_pit
from arviz_plots.plots.utils import process_group_variables_coords
from arviz_plots.plots.utils_plot_types import warn_if_binary, warn_if_prior_predictive


def plot_ppc_pit(
    dt,
    *,
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
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
    r"""PIT Δ-ECDF values with simultaneous confidence envelope.

    For a calibrated model the Probability Integral Transform (PIT) values,
    $p(\tilde{y}_i \le y_i \mid y)$, should be uniformly distributed.
    Where $y_i$ represents the observed data for index $i$ and $\tilde y_i$ represents
    the posterior predictive sample at index $i$.

    This plot shows the empirical cumulative distribution function (ECDF) of the PIT values.
    To make the plot easier to interpret, we plot the Δ-ECDF, that is, the difference between
    the observed ECDF and the expected CDF.
    The points that contribute the most to deviations from uniformity are
    computed as described in [1]_ and highlighted in the plot.

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
    group : str,
        Group to be plotted. Defaults to "posterior_predictive".
        It could also be "prior_predictive".
    coords : dict, optional
        Coordinates to plot.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    method : {"pot_c", "prit_c", "piet_c", "envelope"}, optional
        Method to use for the uniformity test. Defaults to "pot_c".
        Check the documentation of :func:`~arviz_plots.plot_ecdf_pit` for
        more details.
    envelope_prob : float, optional
        Indicates the probability that should be contained within the envelope.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    coverage : bool, optional
        If True, plot the coverage of the central posterior credible intervals. Defaults to False.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

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

        * ecdf_pit -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`. Default is
          ``{"n_simulations": 1000}``.


    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    See Also
    --------
    plot_loo_pit : Predictive check using LOO-PIT Δ-ECDF uniformity test.
    plot_ppc_dist :  Predictive check using 1D marginals for predictive (and observed data).
    plot_ppc_pava : Predictive check ideal for binary, ordinal or categorical data.
    plot_ppc_rootogram : Predictive check ideal for discrete (count) data.

    Examples
    --------
    Plot the ecdf-PIT for the crabs hurdle-negative-binomial dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_pit, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('crabs_hurdle_nb')
        >>> plot_ppc_pit(dt)


    Plot the coverage for the crabs hurdle-negative-binomial dataset.

    .. plot::
        :context: close-figs

        >>> plot_ppc_pit(dt, coverage=True)

    .. minigallery:: plot_ppc_pit

    References
    ----------
    .. [1] Tasso et al. *LOO-PIT predictive model checking* arXiv:2603.02928 (2026).
    """
    envelope_prob = validate_or_use_rcparam(envelope_prob, "stats.envelope_prob")
    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_ppc_pit, "aes_by_visuals"))
    visuals = validate_dict_argument(visuals, (plot_ppc_pit, "visuals"))
    stats = validate_dict_argument(stats, (plot_ppc_pit, "stats"))

    predictive_dist = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    sample_dims = validate_sample_dims(sample_dims, data=predictive_dist)
    observed_dist = process_group_variables_coords(
        dt, group="observed_data", var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    warn_if_binary(observed_dist, predictive_dist)
    warn_if_prior_predictive(group)

    if method not in {"envelope", "pot_c", "prit_c", "piet_c"}:
        raise ValueError(
            f"Method {method} not supported. Choose from 'envelope', 'pot_c', 'prit_c' or 'piet_c'."
        )

    pareto_pit = method in ["pot_c", "piet_c"]

    new_dt = _ppc_pit(predictive_dist, observed_dist, sample_dims, coverage, pareto_pit)

    visuals.setdefault("ylabel", {})
    visuals.setdefault("remove_axis", False)
    visuals.setdefault("xlabel", {"text": "ETI %" if coverage else "PIT"})

    plot_collection = plot_ecdf_pit(
        new_dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group="ecdf_pit",
        coords=coords,
        sample_dims=new_dt.ecdf_pit.dims,
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


def _ppc_pit(predictive_dist, observed_dist, sample_dims, coverage, pareto_pit):
    """Compute Pareto-smoothed PIT ECDF values.

    The probability of the posterior predictive being less than or equal to the observed data
    should be uniformly distributed. This function computes the PIT values with
    Generalized Pareto Distribution tail refinement.

    Parameters
    ----------
    predictive_dist : xarray.Dataset
        The posterior predictive distribution.
    observed_dist : xarray.Dataset
        The observed data.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce.
    coverage : bool
        Whether to compute the coverage.
    pareto_pit : bool
        Whether to use Pareto-smoothed PIT values.
    """
    rng = np.random.default_rng(214)

    dictio = {}
    for var in observed_dist.data_vars:
        if pareto_pit:
            pred_stacked = predictive_dist[var].stack(__sample__=sample_dims)
            vals = xr.apply_ufunc(
                array_stats._pareto_pit_vec,  # pylint: disable=protected-access
                pred_stacked,
                observed_dist[var],
                input_core_dims=[["__sample__"], []],
                output_core_dims=[[]],
                vectorize=False,
                kwargs={"rng": rng},
            )
        else:
            vals_less = (predictive_dist[var] < observed_dist[var]).mean(sample_dims)
            vals_eq = (predictive_dist[var] == observed_dist[var]).mean(sample_dims)
            urvs = rng.uniform(size=vals_less.values.shape)
            vals = vals_less + urvs * vals_eq

        if coverage:
            vals = 2 * np.abs(vals - 0.5)

        dictio[var] = vals

    return xr.DataTree.from_dict({"ecdf_pit": xr.Dataset(dictio)})
