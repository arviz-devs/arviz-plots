"""Predictive intervals plot."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)
from arviz_stats.loo import loo_expectations

from arviz_plots.plots.ppc_interval_plot import _plot_interval
from arviz_plots.plots.utils import process_group_variables_coords


def plot_loo_interval(
    dt,
    *,
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    point_estimate=None,
    ci_kind=None,
    ci_probs=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "trunk",
            "twig",
            "observed_markers",
            "prediction_markers",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str] | bool,
    ] = None,
    visuals: Mapping[
        Literal[
            "trunk",
            "twig",
            "observed_markers",
            "prediction_markers",
            "xlabel",
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[
        Literal["trunk", "twig", "point_estimate"], Mapping[str, Any] | xr.Dataset
    ] = None,
    **pc_kwargs,
):
    """Plot LOO posterior predictive intervals with observed data overlaid.

    Displays observed data as a point and LOO predicted data as a point estimate plus two
    credible intervals.

    Parameters
    ----------
    dt : DataTree
        Input data. It should contain the ``posterior_predictive``, the
        ``log_likelihood`` and ``observed_data`` groups.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, "like", "regex"}, default=None
        If None, interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : str
        Group to be plotted. Defaults to "posterior_predictive".
    coords : dict, optional
        Coordinates of `var_names` to be plotted.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    point_estimate : {"mean", "median"}, optional
        Which point estimate to plot for the predictive distribution.
        Defaults to rcParam ``stats.point_estimate``.
    ci_probs : (float, float), optional
        Indicates the probabilities for the inner (twig) and outer (trunk) credible intervals.
        Defaults to ``(0.5, rcParams["stats.ci_prob"])``. It's assumed that
        ``ci_probs[0] < ci_probs[1]``, otherwise they are sorted.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly", "none"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str or False}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * trunk, twig -> passed to :func:`~arviz_plots.visuals.ci_bound_y`
        * observed_markers -> passed to :func:`~arviz_plots.visuals.point_y`
        * prediction_markers -> passed to :func:`~arviz_plots.visuals.point_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title` defaults to False

    stats : mapping, optional
        Valid keys are:

        * trunk, twig -> passed to loo_expectations for the trunk and twig intervals, respectively
        * point_estimate -> passed to loo_expectations

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    See Also
    --------
    plot_ppc_interval : Plot prior/posterior predictive intervals and observed data.
    plot_ppc_dist : Plot 1D marginals for the posterior/prior predictive and observed data.
    plot_ppc_rootogram : Plot ppc rootogram for discrete (count) data

    Examples
    --------
    Plot LOO posterior predictive intervals for the radon dataset, with custom styling.

    .. plot::
        :context: close-figs

        >>> from arviz_base import load_arviz_data
        >>> import arviz_plots as azp
        >>> azp.style.use("arviz-variat")
        >>> data = load_arviz_data("rugby")
        >>> pc = azp.plot_loo_interval(data)

    .. minigallery:: plot_loo_interval
    """
    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_loo_interval, "aes_by_visuals"))
    visuals = validate_dict_argument(visuals, (plot_loo_interval, "visuals"))
    stats = validate_dict_argument(stats, (plot_loo_interval, "stats"))
    ci_kind = validate_or_use_rcparam(ci_kind, "stats.ci_kind")
    point_estimate = validate_or_use_rcparam(point_estimate, "stats.point_estimate")
    if ci_probs is None:
        rc_ci_prob = rcParams["stats.ci_prob"]
        ci_probs = (0.5, rc_ci_prob)

    ci_probs = np.array(ci_probs)
    if ci_probs.size != 2:
        raise ValueError("ci_probs must have two elements for twig and trunk intervals.")
    if np.any(ci_probs < 0) or np.any(ci_probs > 1):
        raise ValueError("ci_probs must be between 0 and 1.")
    ci_probs.sort()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if labeller is None:
        labeller = BaseLabeller()

    visuals.setdefault("title", False)

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    ds_predictive = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    sample_dims = validate_sample_dims(sample_dims, data=ds_predictive)

    # Extract observed data
    if "observed_data" in dt:
        observed_data = process_group_variables_coords(
            dt,
            group="observed_data",
            var_names=var_names,
            filter_vars=filter_vars,
            coords=coords,
        )
    else:
        observed_data = None

    probs_trunk = [(1 - ci_probs[1]) / 2, (1 + ci_probs[1]) / 2]
    probs_twig = [(1 - ci_probs[0]) / 2, (1 + ci_probs[0]) / 2]
    ci_trunk = (
        loo_expectations(
            dt,
            kind="quantile",
            probs=probs_trunk,
            sample_dims=sample_dims,
            **stats.get("trunk", {}),
        )[0]
        .rename({"quantile": "ci_bound"})
        .assign_coords(ci_bound=["lower", "upper"])
    )
    ci_twig = (
        loo_expectations(
            dt, kind="quantile", probs=probs_twig, sample_dims=sample_dims, **stats.get("twig", {})
        )[0]
        .rename({"quantile": "ci_bound"})
        .assign_coords(ci_bound=["lower", "upper"])
    )

    pe_stats = stats.get("point_estimate", {})
    if point_estimate == "median":
        point, _ = loo_expectations(dt, kind="median", sample_dims=sample_dims, **pe_stats)
    elif point_estimate == "mean":
        point, _ = loo_expectations(dt, kind="mean", sample_dims=sample_dims, **pe_stats)
    else:
        raise ValueError(
            f"point_estimate must be 'mean' or 'median', but {point_estimate} was passed."
        )

    return _plot_interval(
        ds_predictive,
        ci_trunk,
        ci_twig,
        point,
        observed_data,
        sample_dims,
        group,
        plot_collection,
        backend,
        pc_kwargs,
        plot_bknd,
        visuals,
        aes_by_visuals,
        labeller,
    )
