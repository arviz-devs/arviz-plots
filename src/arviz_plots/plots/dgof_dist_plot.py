"""dgof_dist plot code."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import (
    validate_dict_argument,
    validate_or_use_rcparam,
    validate_sample_dims,
)

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dgof_plot import plot_dgof
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    get_visual_kwargs,
    process_group_variables_coords,
    set_grid_layout,
)


def plot_dgof_dist(
    dt,
    *,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind=None,
    method="pot_c",
    envelope_prob=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal["dist", "ecdf_lines", "credible_interval", "title", "xlabel", "ylabel"],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "dist",
            "ecdf_lines",
            "credible_interval",
            "title",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[Literal["dist", "ecdf_pit"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    """Plot 1D marginal distributions and a Δ-ECDF-PIT diagnostic.

    The marginal distributions are plotted using the specified `kind` (kde, histogram, or quantile
    dot plot). Additionally, a Δ-ECDF-PIT diagnostic is plotted to assess the goodness-of-fit of
    the estimated distributions to the underlying data [1]_. If the estimated distributions are
    accurate, the PIT values should be uniformly distributed on [0, 1], resulting in a Δ-ECDF close
    to zero.
    The points that contribute the most to deviations from uniformity are
    computed as described in [2]_.

    Parameters
    ----------
    dt : DataTree
        Input data
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, "like", "regex"}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
        Coordinates to be used to index data variables.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "dot"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    method : {"pot_c", "prit_c", "piet_c", "envelope"}, optional
        Method to use for the uniformity test. Defaults to "pot_c". Check the documentation of
        :func:`~arviz_plots.plot_ecdf_pit` for more details.
    envelope_prob : float, optional
        If `method` is "envelope", indicates the probability that should be contained within the
        envelope, otherwise indicates the probability threshold to highlight points.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection` when
        plotted. Valid keys are the same as for `visuals`.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * dist -> depending on the value of `kind` passed to:

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "hist" -> passed to :func: `~arviz_plots.visuals.step_hist`
          * "dot" -> passed to :func:`~arviz_plots.visuals.scatter_xy`

        * credible_interval -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * ecdf_lines -> passed to :func:`~arviz_plots.visuals.line_xy`
        * title -> passed to :func:`~arviz_plots.visuals.title`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`


    stats : mapping, optional
        Valid keys are:

        * dist -> passed to kde, ecdf and qds for both dist plot and dgof plot
        * ecdf_pit -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit` or
          :func:`~xarray.Dataset.azstats.uniformity_test` depending on the value of `method`.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Default plot with quantile dot marginals and Δ-ECDF-PIT diagnostic:

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_dgof_dist, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data("centered_eight")
        >>> plot_dgof_dist(dt, var_names=["mu" , "tau"], kind="dot");

    .. minigallery:: plot_dgof_dist

    References
    ----------
    .. [1] Säilynoja et al. *Recommendations for visual predictive checks in Bayesian workflow*.
        (2025) arXiv preprint https://arxiv.org/abs/2503.01509

    .. [2] Tasso et al. *LOO-PIT predictive model checking* arXiv:2603.02928 (2026).
    """
    envelope_prob = validate_or_use_rcparam(envelope_prob, "stats.envelope_prob")
    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_dgof_dist, "aes_by_visuals"))
    visuals = validate_dict_argument(visuals, (plot_dgof_dist, "visuals"))
    stats = validate_dict_argument(stats, (plot_dgof_dist, "stats"))

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    sample_dims = validate_sample_dims(sample_dims, data=distribution)
    kind = validate_or_use_rcparam(kind, "plot.density_kind")
    if kind == "auto":
        kind = "kde" if all(da.dtype.kind == "f" for da in distribution.values) else "hist"
    if kind not in ("kde", "hist", "dot"):
        raise ValueError("For plot_dgof_dist kind must be either 'kde', 'hist', 'dot'")

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", ["column"])
        aux_dim_list = [dim for dim in distribution.dims if dim not in sample_dims]
        pc_kwargs.setdefault("rows", ["__variable__"] + aux_dim_list)
        pc_kwargs = set_grid_layout(pc_kwargs, plot_bknd, distribution, num_cols=2)

        plot_collection = PlotCollection.grid(
            distribution.expand_dims(column=2).assign_coords(column=["dist", "gof"]),
            backend=backend,
            **pc_kwargs,
        )

    if labeller is None:
        labeller = BaseLabeller()

    # dens
    visuals_dist = {
        key: False for key in ("credible_interval", "point_estimate", "point_estimate_text")
    }
    visuals_dist["dist"] = get_visual_kwargs(visuals, "dist")
    visuals_dist["title"] = get_visual_kwargs(visuals, "title")

    plot_collection.coords = {"column": "dist"}
    plot_dist(
        dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        kind=kind,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_by_visuals={key: value for key, value in aes_by_visuals.items() if key == "dist"},
        visuals=visuals_dist,
        stats={"dist": stats.get("dist", {})},
    )
    plot_collection.coords = None

    # dgof
    visuals_dgof = get_visual_kwargs(visuals, "dgof")
    visuals_dgof["title"] = get_visual_kwargs(visuals, "title")
    visuals_dgof["ecdf_lines"] = get_visual_kwargs(visuals, "ecdf_lines")
    visuals_dgof["credible_interval"] = get_visual_kwargs(visuals, "credible_interval")
    visuals_dgof["xlabel"] = get_visual_kwargs(visuals, "xlabel")
    visuals_dgof["ylabel"] = get_visual_kwargs(visuals, "ylabel")

    stats_dgof = {"ecdf_pit": stats.get("ecdf_pit", {})}
    stats_dgof["dist"] = stats.get("dist", {})

    plot_collection.coords = {"column": "gof"}
    plot_dgof(
        dt,
        method=method,
        envelope_prob=envelope_prob,
        kind=kind,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_by_visuals={key: value for key, value in aes_by_visuals.items() if key == "dgof"},
        visuals=visuals_dgof,
        stats=stats_dgof,
    )

    return plot_collection
