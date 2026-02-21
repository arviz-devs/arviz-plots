"""pit density diagnostic."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import arviz_stats  # pylint: disable=unused-import
import xarray as xr
from arviz_base import rcParams
from arviz_stats.ecdf_utils import (
    compute_pit_for_histogram,
    compute_pit_for_kde,
    compute_pit_for_qds,
)

from arviz_plots.plots.ecdf_plot import plot_ecdf_pit
from arviz_plots.plots.utils import process_group_variables_coords


def plot_dgof(
    dt,
    *,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind=None,
    envelope_prob=None,
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
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[Literal["dist", "ecdf_pit"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    """Plot a Δ-ECDF-PIT diagnostic for 1D marginal distributions.

    A Δ-ECDF-PIT diagnostic is plotted to assess the goodness-of-fit of the
    estimated distributions to the underlying data using the specified `kind` (kde, histogram,
    or quantile dot plot) [1]_. If the estimated distributions are accurate, the PIT values
    should be uniformly distributed on [0, 1], resulting in a Δ-ECDF close to zero.
    Simultaneous confidence bands are computed using simulation method described in [2]_.

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
        Which method to diagnose the distribution fit.
        Defaults to ``rcParams["plot.density_kind"]``
    envelope_prob : float, optional
        Indicates the probability that should be contained within the envelope.
        Defaults to ``rcParams["stats.envelope_prob"]``.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection` when
        plotted. Valid keys are the same as for `visuals`.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * credible_interval -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * ecdf_lines -> passed to :func:`~arviz_plots.visuals.line_xy`
        * title -> passed to :func:`~arviz_plots.visuals.title`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`


    stats : mapping, optional
        Valid keys are:

        * ecdf_pit -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`.
          Default is ``{"n_simulations": 1000}``.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Δ-ECDF-PIT diagnostic for quantile dot marginals:

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_dgof, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data("centered_eight")
        >>> plot_dgof(dt, var_names=["mu" , "tau"], kind="dot");

    .. minigallery:: plot_dgof


    References
    ----------
    .. [1] Säilynoja et al. *Recommendations for visual predictive checks in Bayesian workflow*.
        (2025) arXiv preprint https://arxiv.org/abs/2503.01509

    .. [2] Säilynoja et al. *Graphical test for discrete uniformity and
       its applications in goodness-of-fit evaluation and multiple sample comparison*.
       Statistics and Computing 32(32). (2022) https://doi.org/10.1007/s11222-022-10090-6
    """
    if envelope_prob is None:
        envelope_prob = rcParams["stats.envelope_prob"]
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if stats is None:
        stats = {}
    else:
        stats = stats.copy()
    if kind is None:
        kind = rcParams["plot.density_kind"]

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if kind == "hist":
        hist_dt = distribution.azstats.histogram(**stats.get("dist", {}))
        new_dt = compute_pit_for_histogram(distribution, hist_dt)
    elif kind == "kde":
        kde_dt = distribution.azstats.kde(**stats.get("dist", {}))
        new_dt = compute_pit_for_kde(distribution, kde_dt)
    elif kind == "dot":
        qd_dt = distribution.azstats.qds(**stats.get("dist", {}))
        new_dt = compute_pit_for_qds(distribution, qd_dt)
    else:
        raise ValueError(f"Kind {kind} not supported. Choose from 'hist', 'kde', or 'dot'.")

    visuals.setdefault("ylabel", {})
    visuals.setdefault("xlabel", {"text": "PIT"})

    stats_ecdf = {"ecdf_pit": stats.get("ecdf_pit", {})}

    plot_collection = plot_ecdf_pit(
        new_dt,
        coords=coords,
        sample_dims="sample",
        envelope_prob=envelope_prob,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats_ecdf,
        **pc_kwargs,
    )

    return plot_collection
