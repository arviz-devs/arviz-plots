"""Plot ppc pit."""
import warnings
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.ecdf_utils import difference_ecdf_pit
from numpy import unique

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, set_wrap_layout
from arviz_plots.visuals import (
    ecdf_line,
    fill_between_y,
    labelled_title,
    labelled_x,
    labelled_y,
    set_xticks,
)


def plot_ppc_pit(
    dt,
    ci_prob=None,
    coverage=False,
    var_names=None,
    data_pairs=None,
    filter_vars=None,  # pylint: disable=unused-argument
    group="posterior_predictive",
    coords=None,  # pylint: disable=unused-argument
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "ecdf_lines",
            "credible_interval",
            "xlabel",
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
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | Literal[False],
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
    the observed ECDF and the expected CDF. Simultaneous confidence bands are computed using
    the method described in described in [1]_.

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
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to ``rcParams["stats.ci_prob"]``
    coverage : bool, optional
        If True, plot the coverage of the central posterior credible intervals. Defaults to False.
    data_pairs : dict, optional
        Dictionary of keys prior/posterior predictive data and values observed data variable names.
        If None, it will assume that the observed data and the predictive data have
        the same variable name.
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
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * ecdf_lines -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * ci -> passed to :func:`~arviz_plots.visuals.ci_line_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

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
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()

    if stats is None:
        stats = {}
    else:
        stats = stats.copy()

    ecdf_pit_kwargs = stats.get("ecdf_pit", {}).copy()
    ecdf_pit_kwargs.setdefault("n_simulations", 1000)

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    labeller = BaseLabeller()

    if data_pairs is None:
        data_pairs = {var_names: var_names}
    if None in data_pairs.keys():
        data_pairs = dict(zip(dt[group].data_vars, dt.observed_data.data_vars))

    randomized = [
        (dt[group][pred_var].values.dtype.kind == "i")
        or (dt.observed_data[obs_var].values.dtype.kind == "i")
        for pred_var, obs_var in data_pairs.items()
    ]

    if any(randomized):
        if any(
            set(unique(dt.observed_data[var].values)).issubset({0, 1})
            for var in data_pairs.values()
        ):
            warnings.warn(
                "Observed data is binary. Use plot_ppc_pava instead",
                stacklevel=2,
            )

    ds_ecdf = difference_ecdf_pit(
        dt, data_pairs, group, ci_prob, coverage, randomized, **ecdf_pit_kwargs
    )

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    colors = plot_bknd.get_default_aes("color", 1, {})

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", "__variable__")

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, ds_ecdf)

        plot_collection = PlotCollection.wrap(
            ds_ecdf,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    ## ecdf_line
    ecdf_ls_kwargs = copy(visuals.get("ecdf_lines", {}))

    if ecdf_ls_kwargs is not False:
        _, _, ecdf_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ecdf_lines", sample_dims
        )
        ecdf_ls_kwargs.setdefault("color", colors[0])

        plot_collection.map(
            ecdf_line,
            "ecdf_lines",
            data=ds_ecdf,
            ignore_aes=ecdf_ls_ignore,
            **ecdf_ls_kwargs,
        )

    if coverage:
        plot_collection.map(
            set_xticks,
            "ecdf_xticks",
            values=[0, 0.25, 0.5, 0.75, 1],
            labels=["0", "25", "50", "75", "100"],
            store_artist=backend == "none",
        )

    ci_kwargs = copy(visuals.get("credible_interval", {}))
    _, _, ci_ignore = filter_aes(plot_collection, aes_by_visuals, "credible_interval", sample_dims)
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", "black")
        ci_kwargs.setdefault("alpha", 0.1)

        plot_collection.map(
            fill_between_y,
            "credible_interval",
            data=ds_ecdf,
            x=ds_ecdf.sel(plot_axis="x"),
            y_bottom=ds_ecdf.sel(plot_axis="y_bottom"),
            y_top=ds_ecdf.sel(plot_axis="y_top"),
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "xlabel", sample_dims
    )
    xlabel_kwargs = copy(visuals.get("xlabel", {}))
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "black")

        if coverage:
            xlabel_kwargs.setdefault("text", "ETI %")
        else:
            xlabel_kwargs.setdefault("text", "PIT")

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=xlabels_ignore,
            subset_info=True,
            **xlabel_kwargs,
        )

    # set ylabel
    _, ylabels_aes, ylabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "ylabel", sample_dims
    )
    ylabel_kwargs = copy(visuals.get("ylabel", {}))
    if ylabel_kwargs is not False:
        if "color" not in ylabels_aes:
            ylabel_kwargs.setdefault("color", "black")

        ylabel_kwargs.setdefault("text", "Δ ECDF")

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=ylabels_ignore,
            subset_info=True,
            **ylabel_kwargs,
        )

    # title
    title_kwargs = copy(visuals.get("title", {}))
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

    return plot_collection
