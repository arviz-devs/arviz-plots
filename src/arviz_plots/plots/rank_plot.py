"""Plot fractional rank."""
from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.ecdf_utils import ecdf_pit

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords, set_wrap_layout
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
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
    stats_kwargs=None,
):
    """Fractional rank Δ-ECDF plots.

    Rank plots are built by replacing the posterior draws by their ranking computed over all chains.
    Then each chain is plotted independently. If all of the chains are targeting the same posterior,
    we expect the ranks in each chain to be uniformly distributed.
    To simplify comparison we compute the ordered fractional ranks, which are distributed
    uniformly in [0, 1]. Additionally, we plot the Δ-ECDF, that is, the difference between the
    expected CDF from the observed ECDF.
    Simultaneous confidence bands are computed using the method described in [1]_.

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
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * ecdf_lines -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * ci -> passed to :func:`~arviz_plots.visuals.ci_line_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats_kwargs : mapping, optional
        Valid keys are:

        * n_simulations -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`. Default is 1000.
        * method -> passed to :func:`~arviz_stats.ecdf_utils.ecdf_pit`. Default is "simulation".

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.grid`

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
    .. [1] Säilynoja T, Bürkner PC. and Vehtari A. *Graphical test for discrete uniformity and
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
    if stats_kwargs is None:
        stats_kwargs = {}
    stats_kwargs.setdefault("n_simulations", 1000)
    stats_kwargs.setdefault("method", "simulation")
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    plot_kwargs.setdefault("remove_axis", True)
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    labeller = BaseLabeller()

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    ecdf_dims = ["draw"]

    # Compute ranks
    dt_ecdf_ranks = distribution.azstats.compute_ranks(dims=sample_dims)
    # Compute ECDF
    dt_ecdf = dt_ecdf_ranks.azstats.ecdf(dims=ecdf_dims, pit=True)

    # Compute envelope
    # This asumes independence between the ranks
    # But we should consider the jointly rank-transformed values
    dummy_vals_size = np.prod([len(distribution[dims]) for dims in ecdf_dims])
    dummy_vals = np.linspace(0, 1, dummy_vals_size)
    print(stats_kwargs)
    x_ci, _, lower_ci, upper_ci = ecdf_pit(dummy_vals, ci_prob, **stats_kwargs)
    lower_ci = lower_ci - x_ci
    upper_ci = upper_ci - x_ci

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in dt_ecdf_ranks.dims if dim not in sample_dims]
        )
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("color", ["chain"])
            pc_kwargs["aes"].setdefault("overlay", ["chain"])

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, dt_ecdf_ranks)
        pc_kwargs["plot_grid_kws"].setdefault("sharex", True)
        plot_collection = PlotCollection.wrap(
            dt_ecdf_ranks,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("ecdf_lines", plot_collection.aes_set)

    ## ecdf_line
    ecdf_ls_kwargs = copy(plot_kwargs.get("ecdf_lines", {}))

    if ecdf_ls_kwargs is not False:
        _, _, ecdf_ls_ignore = filter_aes(plot_collection, aes_map, "ecdf_lines", sample_dims)

        plot_collection.map(
            ecdf_line,
            "ecdf_lines",
            data=dt_ecdf,
            ignore_aes=ecdf_ls_ignore,
            **ecdf_ls_kwargs,
        )

    ci_kwargs = copy(plot_kwargs.get("ci", {}))
    _, _, ci_ignore = filter_aes(plot_collection, aes_map, "ci", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", "black")
        ci_kwargs.setdefault("alpha", 0.1)

        plot_collection.map(
            fill_between_y,
            "ci",
            data=dt_ecdf,
            x=x_ci,
            y_bottom=lower_ci,
            y_top=upper_ci,
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = copy(plot_kwargs.get("xlabel", {}))
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "black")

        xlabel_kwargs.setdefault("text", "Fractional ranks")
        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=xlabels_ignore,
            subset_info=True,
            **xlabel_kwargs,
        )

    # title
    title_kwargs = copy(plot_kwargs.get("title", {}))
    _, _, title_ignore = filter_aes(plot_collection, aes_map, "title", sample_dims)

    if title_kwargs is not False:
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    if plot_kwargs.get("remove_axis", True) is not False:
        plot_collection.map(
            remove_axis,
            store_artist=backend == "none",
            axis="y",
            ignore_aes=plot_collection.aes_set,
        )

    return plot_collection
