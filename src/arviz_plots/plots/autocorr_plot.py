"""Autocorrelation plot code."""

from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords, set_figure_layout
from arviz_plots.visuals import fill_between_y, labelled_title, labelled_x, line, line_xy


def plot_autocorr(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    max_lag=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
):
    """Autocorrelation plots for the given dataset.

    Line plot of the autocorrelation function (ACF)

    The ACF plots can be used as a convergence diagnostic for posteriors from MCMC
    samples.

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
    max_lag : int, optional
        Maximum lag to compute the ACF. Defaults to 100.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * lines -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * ref_line -> passed to :func:`~arviz_plots.visuals.line_xy`
        * ci -> passed to :func:`~arviz_plots.visuals.fill_between_y`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Autocorrelation plot for mu variable in the centered eight dataset.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_autocorrelation, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_autocorrelation(dt, var_names=["mu"])


    .. minigallery:: plot_autocorrelation

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
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

    # Default max lag to 100
    if max_lag is None:
        max_lag = 100

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    acf_dataset = distribution.azstats.autocorr(dims=sample_dims).sel(draw=slice(0, max_lag - 1))
    c_i = 1.96 / acf_dataset.sizes["draw"] ** 0.5
    x_ci = np.arange(0, max_lag)

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    default_linestyle = plot_bknd.get_default_aes("linestyle", 2, {})[1]

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]
            + [dim for dim in distribution.dims if dim not in {"model"}.union(sample_dims)],
        )
        pc_kwargs.setdefault("rows", None)

        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("color", ["chain"])
            pc_kwargs["aes"].setdefault("overlay", ["chain"])

        pc_kwargs = set_figure_layout(pc_kwargs, plot_bknd, distribution)
        pc_kwargs["plot_grid_kws"].setdefault("sharex", True)
        pc_kwargs["plot_grid_kws"].setdefault("sharey", True)

        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("lines", plot_collection.aes_set)

    ## reference line
    ref_ls_kwargs = copy(plot_kwargs.get("ref_line", {}))

    if ref_ls_kwargs is not False:
        _, _, ac_ls_ignore = filter_aes(plot_collection, aes_map, "ref_line", sample_dims)
        ref_ls_kwargs.setdefault("color", "gray")
        ref_ls_kwargs.setdefault("linestyle", default_linestyle)

        plot_collection.map(
            line_xy,
            "ref_line",
            data=acf_dataset,
            x=x_ci,
            y=0,
            ignore_aes=ac_ls_ignore,
            **ref_ls_kwargs,
        )

    ## autocorrelation line
    acf_ls_kwargs = copy(plot_kwargs.get("lines", {}))

    if acf_ls_kwargs is not False:
        _, _, ac_ls_ignore = filter_aes(plot_collection, aes_map, "lines", sample_dims)

        plot_collection.map(
            line,
            "lines",
            data=acf_dataset,
            ignore_aes=ac_ls_ignore,
            **acf_ls_kwargs,
        )

    # Plot confidence intervals
    ci_kwargs = copy(plot_kwargs.get("ci", {}))
    _, _, ci_ignore = filter_aes(plot_collection, aes_map, "ci", "draw")
    if ci_kwargs is not False:
        ci_kwargs.setdefault("color", "black")
        ci_kwargs.setdefault("alpha", 0.1)

        plot_collection.map(
            fill_between_y,
            "ci",
            data=acf_dataset,
            x=x_ci,
            y=0,
            y_bottom=-c_i,
            y_top=c_i,
            ignore_aes=ci_ignore,
            **ci_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = copy(plot_kwargs.get("xlabel", {}))
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "black")

        xlabel_kwargs.setdefault("text", "Lag")
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

    return plot_collection
