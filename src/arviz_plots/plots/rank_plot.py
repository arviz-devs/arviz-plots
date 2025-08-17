"""Plot fractional rank."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.ecdf_utils import ecdf_pit

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_contrast_colors,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import ecdf_line, fill_between_y, labelled_title, labelled_x, remove_axis


def plot_rank(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    ci_prob=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
            "title",
            "remove_axis",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[Literal["ecdf_pit"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    """Fractional rank Δ-ECDF plots.

    Rank plots are built by replacing the posterior draws by their ranking computed over all chains.
    Then each chain is plotted independently. If all of the chains are targeting the same posterior,
    we expect the ranks in each chain to be uniformly distributed.
    To simplify comparison we compute the ordered fractional ranks, which are distributed
    uniformly in [0, 1]. Additionally, we plot the Δ-ECDF, that is, the difference between the
    expected CDF from the observed ECDF.
    Simultaneous confidence bands are computed using the simulation method described in [1]_.

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
    group : str, optional
        Which group to use. Defaults to "posterior".
    coords : dict, optional
        Coordinates to plot.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to ``rcParams["stats.ci_prob"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * ecdf_lines -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * credible_interval -> passed to :func:`~arviz_plots.visuals.ci_line_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats : mapping, optional
        Valid keys are:

        * ecdf_pit -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`. Default is
          ``{"n_simulation": 1000}``.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Rank plot for the crabs hurdle-negative-binomial dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_rank, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('crabs_hurdle_nb')
        >>> plot_rank(dt, var_names=["~mu"])


    .. minigallery:: plot_rank

    References
    ----------
    .. [1] Säilynoja et al. *Graphical test for discrete uniformity and
       its applications in goodness-of-fit evaluation and multiple sample comparison*.
       Statistics and Computing 32(32). (2022) https://doi.org/10.1007/s11222-022-10090-6
    """
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if stats is None:
        stats = {}
    else:
        stats = stats.copy()

    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()
    visuals.setdefault("remove_axis", True)

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    labeller = BaseLabeller()

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    ecdf_pit_kwargs = stats.get("ecdf_pit", {}).copy()
    ecdf_pit_kwargs.setdefault("n_simulations", 1000)
    ecdf_pit_kwargs.setdefault("n_chains", distribution.sizes["chain"])
    ecdf_dims = ["draw"]

    # Compute ranks
    dt_ecdf_ranks = distribution.azstats.compute_ranks(dim=sample_dims)
    # Compute ECDF
    dt_ecdf = dt_ecdf_ranks.azstats.ecdf(dim=ecdf_dims, pit=True)

    # Compute envelope
    # This asumes independence between the ranks
    # But we should consider the jointly rank-transformed values
    dummy_vals_size = np.prod([len(distribution[dims]) for dims in ecdf_dims])
    dummy_vals = np.linspace(0, 1, dummy_vals_size)
    x_ci, _, lower_ci, upper_ci = ecdf_pit(dummy_vals, ci_prob, **ecdf_pit_kwargs)
    lower_ci = lower_ci - x_ci
    upper_ci = upper_ci - x_ci

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    bg_color = plot_bknd.get_background_color()
    contrast_color = get_contrast_colors(bg_color=bg_color)

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("col_wrap", 4)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in dt_ecdf_ranks.dims if dim not in sample_dims]
        )
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("color", ["chain"])
            pc_kwargs["aes"].setdefault("overlay", ["chain"])

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, dt_ecdf_ranks)
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)
        plot_collection = PlotCollection.wrap(
            dt_ecdf_ranks,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals.setdefault("ecdf_lines", plot_collection.aes_set)

    ## ecdf_line
    ecdf_ls_kwargs = get_visual_kwargs(visuals, "ecdf_lines")

    if ecdf_ls_kwargs is not False:
        _, _, ecdf_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ecdf_lines", sample_dims
        )

        plot_collection.map(
            ecdf_line,
            "ecdf_lines",
            data=dt_ecdf,
            ignore_aes=ecdf_ls_ignore,
            **ecdf_ls_kwargs,
        )

    ci_kwargs = get_visual_kwargs(visuals, "credible_interval")
    _, _, ci_ignore = filter_aes(plot_collection, aes_by_visuals, "credible_interval", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", contrast_color)
        ci_kwargs.setdefault("alpha", 0.1)

        plot_collection.map(
            fill_between_y,
            "credible_interval",
            data=dt_ecdf,
            x=x_ci,
            y_bottom=lower_ci,
            y_top=upper_ci,
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "xlabel", sample_dims
    )
    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", contrast_color)

        xlabel_kwargs.setdefault("text", "Fractional ranks")
        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=xlabels_ignore,
            subset_info=True,
            **xlabel_kwargs,
        )

    # title
    title_kwargs = get_visual_kwargs(visuals, "title")
    _, _, title_ignore = filter_aes(plot_collection, aes_by_visuals, "title", sample_dims)

    if title_kwargs is not False:
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    if visuals.get("remove_axis", True) is not False:
        plot_collection.map(
            remove_axis,
            store_artist=backend == "none",
            axis="y",
            ignore_aes=plot_collection.aes_set,
        )

    return plot_collection
