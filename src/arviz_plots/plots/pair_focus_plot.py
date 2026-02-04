"""Pair focus plot code."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import labelled_x, labelled_y, scatter_x, scatter_xy


def plot_pair_focus(
    dt,
    focus_var,
    *,
    focus_var_coords=None,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "scatter",
            "divergence",
            "xlabel",
            "ylabel",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "scatter",
            "divergence",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    **pc_kwargs,
):
    """Plot a fixed variable against other variables in the dataset.

    Parameters
    ----------
    dt : DataTree
        Input data
    focus_var: str or DataArray
        Name of the variable or DataArray to be plotted against all other variables.
    focus_var_coords : mapping, optional
        Coordinates to use for the target variable.
    var_names: str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars: {None, “like”, “regex”}, default None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to use for plotting. Defaults to "posterior".
    coords : mapping, optional
        Coordinates to use for plotting.
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly", "none"}, optional
        Plotting backend to use. Defaults to ``rcParams["plot.backend"]``
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.
        By default, there are no aesthetic mappings at all
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * scatter -> passed to :func:`~.visuals.scatter_x`
        * divergence -> passed to :func:`~.visuals.scatter_xy`. Defaults to False.
        * xlabel -> :func:`~.visuals.labelled_x`
        * ylabel -> :func:`~.visuals.labelled_y`

    **pc_kwargs
        Passed to :meth:`arviz_plots.PlotCollection.wrap`


    Returns
    -------
    PlotCollection

    Examples
    --------
    Default plot_pair_focus

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair_focus, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair_focus(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>>     focus_var="theta",
        >>>     focus_var_coords={"school": "Choate"},
        >>> )

    .. minigallery:: plot_pair_focus

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

    if isinstance(focus_var, str):
        y = (
            get_group(dt, group)[focus_var].sel(focus_var_coords)
            if focus_var_coords
            else get_group(dt, group)[focus_var]
        )
    elif isinstance(focus_var, xr.DataArray):
        y = focus_var
    else:
        raise TypeError(
            f"focus_var should be a string or DataArray, got {type(focus_var)} instead."
        )

    dims_y = [dim for dim in y.dims if dim not in sample_dims]
    if len(dims_y) > 0:
        raise ValueError(f"focus variable has unexpected dimensions: {dims_y}.")

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

    aes_by_visuals["scatter"] = {"overlay"}.union(aes_by_visuals.get("scatter", {}))
    scatter_kwargs = get_visual_kwargs(visuals, "scatter")
    if scatter_kwargs is not False:
        _, scatter_aes, scatter_ignore = filter_aes(
            plot_collection, aes_by_visuals, "scatter", sample_dims
        )

        if "color" not in scatter_aes:
            scatter_kwargs.setdefault("color", "C0")

        if "width" not in scatter_aes:
            scatter_kwargs.setdefault("width", 0)

        if "alpha" not in scatter_aes:
            scatter_kwargs.setdefault("alpha", 0.5)

        plot_collection.map(
            scatter_x,
            "scatter",
            ignore_aes=scatter_ignore,
            y=y,
            **scatter_kwargs,
        )

    # divergence

    aes_by_visuals["divergence"] = {"overlay"}.union(aes_by_visuals.get("divergence", {}))
    div_kwargs = get_visual_kwargs(visuals, "divergence", False)
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
            div_kwargs.setdefault("color", "C1")
        if "alpha" not in div_aes:
            div_kwargs.setdefault("alpha", 0.4)
        plot_collection.map(
            scatter_xy,
            "divergence",
            ignore_aes=div_ignore,
            y=y,
            mask=divergence_mask,
            **div_kwargs,
        )

    if labeller is None:
        labeller = BaseLabeller()

    # xlabel of plots

    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")

    if xlabel_kwargs is not False:
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

    ylabel_kwargs = get_visual_kwargs(visuals, "ylabel")
    if ylabel_kwargs is not False:
        _, _, ylabel_ignore = filter_aes(plot_collection, aes_by_visuals, "ylabel", sample_dims)

        # generate y label text using labeller
        focus_var_coords = {key: value.item() for key, value in y.coords.items() if value.size <= 1}
        y_label_text = labeller.make_label_vert(
            y.name, focus_var_coords, {name: 0 for name in focus_var_coords}
        )

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=ylabel_ignore,
            text=y_label_text,
            **ylabel_kwargs,
        )

    return plot_collection
