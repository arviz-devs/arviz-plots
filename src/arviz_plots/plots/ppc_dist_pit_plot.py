"""Predictive check using densities and PIT Δ-ECDFs."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import xarray as xr
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument, validate_or_use_rcparam

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.ecdf_plot import plot_ecdf_pit
from arviz_plots.plots.ppc_dist_plot import plot_ppc_dist
from arviz_plots.plots.utils import filter_aes, get_visual_kwargs, set_grid_layout
from arviz_plots.plots.utils_ppc import get_ppc_pit, get_suspicious_mask_ds, prepare_ppc_dist_data
from arviz_plots.visuals import trace_rug


def plot_ppc_dist_pit(
    dt,
    *,
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    kind=None,
    num_samples=50,
    method="pot_c",
    envelope_prob=None,
    coverage=False,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "predictive_dist",
            "observed_dist",
            "ecdf_lines",
            "credible_interval",
            "suspicious_points",
            "p_value_text",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "predictive_dist",
            "observed_dist",
            "suspicious_rug",
            "ecdf_lines",
            "credible_interval",
            "suspicious_points",
            "p_value_text",
            "xlabel_dist",
            "xlabel_pit",
            "ylabel",
            "title",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[
        Literal["predictive_dist", "observed_dist", "ecdf_pit"], Mapping[str, Any] | xr.Dataset
    ] = None,
    **pc_kwargs,
):
    """1D marginals for the predictive distribution and PIT Δ-ECDF.

    The left column shows 1D marginals for the posterior predictive distribution
    overlaid on the observed data, identical to :func:`~arviz_plots.plot_ppc_dist`.

    The right column shows the empirical CDF (ECDF) of the PIT values minus the expected
    CDF, identical to :func:`~arviz_plots.plot_ppc_pit`.

    Suspicious observations are computed from the uniformity test and they are highlighted
    in both columns, either as rug marks at y=0 in the dist column or as points in ECDF for
    the PIT column. The suspicious observations are the ones that contribute the most to
    deviations from uniformity.

    Parameters
    ----------
    dt : DataTree
        Input data with ``posterior_predictive`` and ``observed_data`` groups.
    var_names : str or list of str, optional
        Variables to plot.
    filter_vars : {None, "like", "regex"}, optional
    group : str,
        Group to be plotted. Defaults to "posterior_predictive".
        It could also be "prior_predictive".
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Defaults to ``rcParams["data.sample_dims"]``.
    kind : {"auto", "kde", "hist", "ecdf", "dot"}, optional
        Density kind for the dist column.
        Defaults to ``rcParams["plot.density_kind"]``.
    num_samples : int, default 50
        Number of predictive draws to overlay in the dist column.
    method : {"pot_c", "prit_c", "piet_c", "envelope"}, default "pot_c"
        Uniformity-test method for the PIT column.
    envelope_prob : float, optional
        Probability inside the simultaneous envelope.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    coverage : bool, default False
        If True, replace PIT with ``2|PIT - 0.5|`` to assess ETI coverage.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Valid keys: ``predictive_dist``, ``observed_dist``, ``ecdf_lines``,
        ``credible_interval``, ``suspicious_points``, ``p_value_text``, ``title``.
    visuals : mapping, optional
        Valid keys:

        * predictive_dist  -> density lines for predictive draws
        * observed_dist    -> density line for observed data
        * suspicious_rug   -> rug marks at y=0 for suspicious observations in the dist column
        * ecdf_lines       -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * credible_interval -> only when ``method="envelope"``
        * suspicious_points -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * p_value_text     -> passed to :func:`~arviz_plots.visuals.annotate_xy`
        * xlabel_dist      -> x-axis label for the dist column
        * xlabel_pit       -> x-axis label for the PIT column
        * ylabel           -> y-axis label for the PIT column
        * title            -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis      -> set to ``False`` to skip axis removal

    stats : mapping, optional
        Valid keys: ``predictive_dist``, ``observed_dist``, ``ecdf_pit``.

    **pc_kwargs
        Passed to :class:`~arviz_plots.PlotCollection.grid`.

    Returns
    -------
    PlotCollection

    See Also
    --------
    plot_ppc_dist : Predictive density check only.
    plot_ppc_pit  : PIT Δ-ECDF check only.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_dist_pit, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('crabs_hurdle_nb')
        >>> plot_ppc_dist_pit(dt)

    .. minigallery:: plot_ppc_dist_pit
    """
    envelope_prob = validate_or_use_rcparam(envelope_prob, "stats.envelope_prob")
    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_ppc_dist_pit, "aes_by_visuals"))
    visuals = validate_dict_argument(visuals, (plot_ppc_dist_pit, "visuals"))
    stats = validate_dict_argument(stats, (plot_ppc_dist_pit, "stats"))

    if method not in {"envelope", "pot_c", "prit_c", "piet_c"}:
        raise ValueError(
            f"Method {method!r} not supported. "
            "Choose from 'envelope', 'pot_c', 'prit_c' or 'piet_c'."
        )

    plot_bknd, _, sample_dims, predictive_dist, predictive_dist_sub, observed_dist = (
        prepare_ppc_dist_data(
            dt,
            var_names=var_names,
            filter_vars=filter_vars,
            group=group,
            coords=coords,
            sample_dims=sample_dims,
            kind=kind,
            num_samples=num_samples,
            plot_collection=plot_collection,
            backend=backend,
            stats=stats,
            require_observed=True,
        )
    )

    pareto_pit = method in ("pot_c", "piet_c")
    pit_dt = get_ppc_pit(predictive_dist, observed_dist, sample_dims, coverage, pareto_pit)
    pit_dims = pit_dt.ecdf_pit.dims

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharex", False)
        pc_kwargs["figure_kwargs"].setdefault("sharey", False)

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("overlay_ppc", ["sample"])
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs.setdefault("rows", "__variable__")

        pc_kwargs = set_grid_layout(pc_kwargs, plot_bknd, predictive_dist_sub, num_cols=2)

        plot_collection = PlotCollection.grid(
            predictive_dist_sub.expand_dims(column=2).assign_coords(column=["dist", "pit"]),
            backend=backend,
            **pc_kwargs,
        )

    if labeller is None:
        labeller = BaseLabeller()

    dist_aes_by_visuals = {
        key: value
        for key, value in aes_by_visuals.items()
        if key in ("predictive_dist", "observed_dist", "title")
    }
    dist_visuals = {
        key: value
        for key, value in visuals.items()
        if key in ("predictive_dist", "observed_dist", "title", "remove_axis")
    }
    plot_collection.coords = {"column": "dist"}
    plot_collection = plot_ppc_dist(
        dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        kind=kind,
        num_samples=num_samples,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=dist_aes_by_visuals,
        visuals=dist_visuals,
        stats={
            "predictive_dist": stats.get("predictive_dist", {}),
            "observed_dist": stats.get("observed_dist", {}),
        },
        **pc_kwargs,
    )
    plot_collection.coords = None

    # Plot suspicious obs from uniformity test
    rug_kwargs = get_visual_kwargs(visuals, "suspicious_rug")
    if rug_kwargs is not False and method != "envelope":
        suspicious_mask_ds = get_suspicious_mask_ds(
            envelope_prob, observed_dist, pit_dt, method, stats
        )
        rug_kwargs.setdefault("color", "C1")
        rug_kwargs.setdefault("marker", "|")
        _, _, rug_ignore = filter_aes(
            plot_collection, aes_by_visuals, "suspicious_rug", sample_dims
        )
        plot_collection.map(
            trace_rug,
            "suspicious_rug",
            data=observed_dist,
            mask=suspicious_mask_ds,
            ignore_aes=rug_ignore,
            xname=False,
            y=0,
            coords={"column": "dist"},
            **rug_kwargs,
        )

    pit_visuals = {
        "ylabel": visuals.get("ylabel", {}),
        "remove_axis": False,
        "xlabel": visuals.get("xlabel_pit", {"text": "ETI %" if coverage else "PIT"}),
    }
    for key in ("ecdf_lines", "credible_interval", "suspicious_points", "p_value_text"):
        if key in visuals:
            pit_visuals[key] = visuals[key]

    pit_aes_by_visuals = {
        k: v
        for k, v in aes_by_visuals.items()
        if k in ("ecdf_lines", "credible_interval", "suspicious_points", "p_value_text")
    }

    plot_collection.coords = {"column": "pit"}
    plot_collection = plot_ecdf_pit(
        pit_dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group="ecdf_pit",
        coords=coords,
        sample_dims=pit_dims,
        method=method,
        envelope_prob=envelope_prob,
        coverage=coverage,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=pit_aes_by_visuals,
        visuals=pit_visuals,
        stats={"ecdf_pit": stats.get("ecdf_pit", {})},
    )
    plot_collection.coords = None

    return plot_collection
