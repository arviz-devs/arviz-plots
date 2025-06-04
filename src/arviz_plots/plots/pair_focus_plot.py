"""Pair focus plot code."""
from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection, leaf_dataset, process_facet_dims
from arviz_plots.plots.utils import filter_aes, get_group, process_group_variables_coords
from arviz_plots.visuals import divergence_scatter, labelled_title, scatter_x


def plot_pair_focus(
    dt,
    var_names=None,
    target_name=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    target_coords=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals=None,
    visuals=None,
    **pc_kwargs,
):
    """Plot pair focus plot of a variable against all other variables in the dataset.

    Parameters
    ----------
    dt : DataTree
        Input data
    var_names: str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    target_name: str
        Name of the variable to be plotted against all other variables.
    filter_vars: {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, optional
        Group to use for plotting. Defaults to "posterior".
    coords : mapping, optional
        Coordinates to use for plotting. Defaults to None.
    target_coords : mapping, optional
        Coordinates to use for the target variable. Defaults to None.
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh","plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * sample -> passed to :func:`~.visuals.scatter_x`
        * divergence -> passed to :func:`~.visuals.divergence_scatter`
        * title -> :func:`~.visuals.labelled_title`

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection

    Examples
    --------
    The following examples focus on behaviour specific to ``plot_pair_focus``.
    For a general introduction to batteries-included functions like this one and common
    usage examples see :ref:`plots_intro`

    Default plot_pair_focus

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair_plot, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> centered = load_arviz_data('centered_eight')
        >>> target_name = "mu"
        >>> var_names = ["theta", "tau"]
        >>> pc = plot_trace(
        >>>     centered,
        >>>     var_names=var_names,
        >>>     target_name=target_name,
        >>>     backend="matplotlib",
        >>> )
        >>> pc.show()

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if visuals is None:
        visuals = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        figsize = pc_kwargs.get("figure_kwargs", {}).get("figsize", None)
        figsize_units = pc_kwargs.get("figure_kwargs", {}).get("figsize_units", "dots")
        pc_kwargs.setdefault("col_wrap", 4)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in distribution.dims if dim not in sample_dims]
        )
        n_plots, _ = process_facet_dims(distribution, pc_kwargs["cols"])
        col_wrap = pc_kwargs["col_wrap"]
        if n_plots <= col_wrap:
            n_rows, n_cols = 1, n_plots
        else:
            div_mod = divmod(n_plots, col_wrap)
            n_rows = div_mod[0] + (div_mod[1] != 0)
            n_cols = col_wrap
    else:
        figsize, figsize_units = plot_bknd.get_figsize(plot_collection)
        n_rows = leaf_dataset(plot_collection.viz, "row").max().to_array().max().item()
        n_cols = leaf_dataset(plot_collection.viz, "col").max().to_array().max().item()

    figsize = plot_bknd.scale_fig_size(
        figsize,
        rows=n_rows,
        cols=n_cols,
        figsize_units=figsize_units,
    )

    # scatter-pair-plot

    if plot_collection is None:
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        if "figsize" not in pc_kwargs["figure_kwargs"]:
            pc_kwargs["figure_kwargs"]["figsize"] = figsize
            pc_kwargs["figure_kwargs"]["figsize_units"] = figsize_units
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)
        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    y = (
        dt.posterior[target_name].sel(target_coords)
        if target_coords is not None
        else dt.posterior[target_name]
    )
    aes_by_visuals["sample"] = {"overlay"}.union(aes_by_visuals.get("sample", {}))
    sample_kwargs = copy(visuals.get("sample", {}))
    sample_kwargs.setdefault("alpha", 0.5)
    sample_kwargs.setdefault("color", "#3f90da")
    _, _, sample_ignore = filter_aes(plot_collection, aes_by_visuals, "sample", sample_dims)

    plot_collection.map(
        scatter_x,
        "sample",
        ignore_aes=sample_ignore,
        y=y,
        **sample_kwargs,
    )

    # divergences

    aes_by_visuals["divergence"] = {"overlay"}.union(aes_by_visuals.get("divergence", {}))
    div_kwargs = copy(visuals.get("divergence", {}))
    sample_stats = get_group(dt, "sample_stats", allow_missing=True)
    if (
        div_kwargs is not False
        and sample_stats is not None
        and "diverging" in sample_stats.data_vars
        and np.any(sample_stats.diverging)
    ):
        divergence_mask = dt.sample_stats.diverging
        _, div_aes, div_ignore = filter_aes(
            plot_collection, aes_by_visuals, "divergence", sample_dims
        )
        if "color" not in div_aes:
            div_kwargs.setdefault("color", "black")
        div_kwargs.setdefault("alpha", 0.4)
        plot_collection.map(
            divergence_scatter,
            "divergence",
            ignore_aes=div_ignore,
            y=y,
            mask=divergence_mask,
            **div_kwargs,
        )

        # title of plots

        if labeller is None:
            labeller = BaseLabeller()

        title_kwargs = copy(visuals.get("title", {}))
        _, _, title_ignore = filter_aes(plot_collection, aes_by_visuals, "title", sample_dims)

        plot_collection.map(
            labelled_title,
            "title",
            subset_info=True,
            labeller=labeller,
            ignore_aes=title_ignore,
            size=8,
            **title_kwargs,
        )

    return plot_collection
