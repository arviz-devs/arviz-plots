"""LOO PAV-adjusted calibration plot."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base.validate import validate_dict_argument, validate_sample_dims
from arviz_stats.loo import loo
from scipy.special import logsumexp

from arviz_plots.plots.pava_calibration_plot import plot_ppc_pava
from arviz_plots.plots.utils import _var_names


def plot_loo_pava(
    dt,
    *,
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,  # pylint: disable=unused-argument
    sample_dims=None,
    data_type="binary",
    ci_prob=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "lines",
            "markers",
            "reference_line",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "lines",
            "markers",
            "reference_line",
            "credible_interval",
            "xlabel",
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    **pc_kwargs,
):
    """LOO PAV-adjusted calibration plot.

    Uses PSIS-LOO-CV to resample the posterior predictive distribution, then
    applies the pool adjacent violators (PAV) algorithm for isotonic regression.
    A 45-degree line corresponds to perfect calibration. Details on the PAV-adjusted
    calibration plot are discussed in [1]_ and [2]_, and PSIS-LOO-CV in [3]_ and [4]_.

    Parameters
    ----------
    dt : DataTree
        Input data. It should contain the ``posterior``, ``posterior_predictive``,
        ``log_likelihood`` and ``observed_data`` groups.
    var_names : str or list of str, optional
        One or more variables to be plotted. Currently only one variable is supported.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, "like", "regex"}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : str, optional
        Only "posterior_predictive" is supported.
    coords : dict, optional
        Coordinates to plot. CURRENTLY NOT IMPLEMENTED
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    data_type : str
        Defaults to "binary". Other options are "categorical" and "ordinal".
        If "categorical", the plot will show the "one-vs-others" calibration and generate one plot
        per category. If "ordinal", the plot will display cumulative conditional event
        probabilities and generate (number of categories - 1) plots.
    ci_prob : float, optional
        Probability for the credible interval. Defaults to ``rcParams["stats.ci_prob"]``.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * lines -> passed to :func:`~arviz_plots.visuals.line_xy`
        * markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * reference_line -> passed to :func:`~arviz_plots.visuals.line_xy`
        * credible_interval -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

        markers defaults to False, no markers are plotted.
        Pass an (empty) mapping to plot markers.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    See Also
    --------
    plot_ppc_pava : PAV-adjusted calibration plot using posterior predictive.

    Examples
    --------
    Plot the LOO PAVA calibration plot for the anes dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_loo_pava, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('anes')
        >>> plot_loo_pava(dt, ci_prob=0.90)


    .. minigallery:: plot_loo_pava

    References
    ----------
    .. [1] Säilynoja et al. *Recommendations for visual predictive checks in Bayesian workflow*.
        (2025) arXiv preprint https://arxiv.org/abs/2503.01509

    .. [2] Dimitriadis et al *Stable reliability diagrams for probabilistic classifiers*.
        PNAS, 118(8) (2021). https://doi.org/10.1073/pnas.2016191118

    .. [3] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out
        cross-validation and WAIC*. Statistics and Computing. 27(5) (2017)
        https://doi.org/10.1007/s11222-016-9696-4

    .. [4] Vehtari et al. *Pareto Smoothed Importance Sampling*. Journal of Machine Learning
        Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
    """
    if group != "posterior_predictive":
        raise ValueError(f"Group {group} not supported. Only 'posterior_predictive' is supported.")

    sample_dims = validate_sample_dims(sample_dims, data=dt.posterior_predictive)
    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_loo_pava, "aes_by_visuals"))
    visuals = validate_dict_argument(visuals, (plot_loo_pava, "visuals"))

    var_names = _var_names(var_names, dt.observed_data, filter_vars)
    if var_names is None:
        var_names = list(dt.observed_data.data_vars)

    new_dt = _loo_resample(dt, sample_dims, var_names)

    return plot_ppc_pava(
        new_dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group="posterior_predictive",
        coords=coords,
        sample_dims=None,
        data_type=data_type,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        **pc_kwargs,
    )


def _loo_resample(dt, sample_dims, resolved_var_names):
    """Resample the posterior predictive using PSIS-LOO weights."""
    resampled_pp = {}
    rng = np.random.default_rng(31247)
    for var_name in resolved_var_names:
        log_weights = loo(dt, pointwise=True, var_name=var_name).log_weights

        pp = dt.posterior_predictive[var_name]
        obs_dims = [d for d in pp.dims if d not in sample_dims]
        pp_flat = pp.stack(sample=sample_dims).transpose("sample", *obs_dims)
        lw_flat = log_weights.stack(sample=sample_dims).transpose("sample", *obs_dims)

        lw_vals = lw_flat.values
        n_draws, n_obs = pp_flat.shape
        probs = np.exp(lw_vals - logsumexp(lw_vals, axis=0))

        resamples = np.empty((n_draws, n_obs), dtype=pp.dtype)
        for j in range(n_obs):
            resamples[:, j] = rng.choice(pp_flat[:, j].values, size=n_draws, p=probs[:, j])

        resampled_da = xr.DataArray(
            resamples[None, :, :],
            dims=["chain", "draw", *obs_dims],
            coords={
                "chain": np.arange(1),
                "draw": np.arange(n_draws),
                **{d: pp.coords[d] for d in obs_dims},
            },
            name=var_name,
        )
        resampled_pp[var_name] = resampled_da

    pp_ds = xr.Dataset(resampled_pp)
    obs_ds = xr.Dataset({v: dt.observed_data[v] for v in resolved_var_names})

    new_dt = xr.DataTree.from_dict(
        {
            "posterior_predictive": pp_ds,
            "observed_data": obs_ds,
        }
    )
    return new_dt
