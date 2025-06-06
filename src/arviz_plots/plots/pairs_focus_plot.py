"""Pair focus plot code."""
from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import divergence_scatter, labelled_x, labelled_y, scatter_x


def plot_pairs_focus(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    focus_var=None,
    focus_var_coords=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals=None,
    visuals=None,
    **pc_kwargs,
):
    """Plot a fixed variable against other variables in the dataset.

    Parameters
    ----------
    dt : DataTree
        Input data
    var_names: str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars: {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, optional
        Group to use for plotting. Defaults to "posterior".
    coords : mapping, optional
        Coordinates to use for plotting. Defaults to None.
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    focus_var: str
        Name of the variable to be plotted against all other variables.
    focus_var_coords : mapping, optional
        Coordinates to use for the target variable. Defaults to None.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh","plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * scatter -> passed to :func:`~.visuals.scatter_x`
        * divergence -> passed to :func:`~.visuals.divergence_scatter`. Defaults to False.
        * title -> :func:`~.visuals.labelled_title`

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Default plot_pair_focus

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pairs_focus, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pairs_focus(
        >>>     dt,
        >>>     var_names=["theta", "tau"],
        >>>     focus_var="mu",
        >>> )

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

    if var_names is None:
        var_names = "~" + focus_var

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in distribution.dims if dim not in sample_dims]
        )
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, distribution)
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        pc_kwargs["figure_kwargs"].setdefault("sharey", True)
        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    # scatter
    y = (
        dt.posterior[focus_var].sel(focus_var_coords)
        if focus_var_coords is not None
        else dt.posterior[focus_var]
    )
    aes_by_visuals["scatter"] = {"overlay"}.union(aes_by_visuals.get("scatter", {}))
    scatter_kwargs = copy(visuals.get("scatter", {}))
    scatter_kwargs.setdefault("alpha", 0.5)
    colors = plot_bknd.get_default_aes("color", 1, {})
    scatter_kwargs.setdefault("color", colors[0])
    scatter_kwargs.setdefault("width", 0)
    _, _, scatter_ignore = filter_aes(plot_collection, aes_by_visuals, "scatter", sample_dims)

    plot_collection.map(
        scatter_x,
        "scatter",
        ignore_aes=scatter_ignore,
        y=y,
        **scatter_kwargs,
    )

    # divergence

    aes_by_visuals["divergence"] = {"overlay"}.union(aes_by_visuals.get("divergence", {}))
    div_kwargs = copy(visuals.get("divergence", False))
    if div_kwargs is True:
        div_kwargs = {}
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

    if labeller is None:
        labeller = BaseLabeller()

    # xlabel of plots

    xlabel_kwargs = copy(visuals.get("xlabel", {}))
    _, _, xlabel_ignore = filter_aes(plot_collection, aes_by_visuals, "xlabel", sample_dims)
    plot_collection.map(
        labelled_x,
        "xlabel",
        subset_info=True,
        ignore_aes=xlabel_ignore,
        labeller=labeller,
        **xlabel_kwargs,
    )

    # ylabel of plots
    ylabel_kwargs = copy(visuals.get("ylabel", {}))
    _, _, ylabel_ignore = filter_aes(plot_collection, aes_by_visuals, "ylabel", sample_dims)
    plot_collection.map(
        labelled_y,
        "ylabel",
        ignore_aes=ylabel_ignore,
        text=focus_var,
        **ylabel_kwargs,
    )

    return plot_collection
