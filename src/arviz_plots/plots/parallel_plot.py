"""Pair focus plot code."""
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import dataset_to_dataarray, rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import multiple_lines, set_xticks


def plot_parallel(
    dt,
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
            "line",
            "xticks",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "line",
            "divergence",
            "xticks",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    **pc_kwargs,
):
    """Plot parallel coordinates plot showing posterior points with and without divergences.

    Parameters
    ----------
    dt : DataTree
        Input data
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
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * line -> passed to :func:`~.visuals.multiple_lines`
        * divergence -> not passed anywhere, can only be set ``False`` to avoid divergence plotting.
        * xticks -> passed to :func:`~.visuals.set_xticks`. Defaults to False.


    **pc_kwargs
        Passed to :meth:`arviz_plots.PlotCollection.wrap`


    Returns
    -------
    PlotCollection

    Examples
    --------
    Default plot_parallel

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_parallel, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_parallel(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>> )

    .. minigallery:: plot_parallel

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

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if labeller is None:
        labeller = BaseLabeller()

    data = process_group_variables_coords(
        dt,
        group=group,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
        allow_dict=False,
    )
    if len(sample_dims) > 1:
        data = data.stack(sample=sample_dims)
        combined_dim = "sample"
    else:
        combined_dim = sample_dims[0]

    # create divergence mask
    sample_stats = get_group(dt, "sample_stats", allow_missing=True)
    if (
        sample_stats is not None
        and "diverging" in sample_stats.data_vars
        and np.any(sample_stats.diverging)
    ):
        div_mask = sample_stats.diverging
        if coords is not None:
            div_mask = div_mask.sel(
                {key: value for key, value in coords.items() if key in div_mask.dims}
            )
        if len(sample_dims) > 1:
            div_mask = div_mask.stack(sample=sample_dims)
    else:
        div_mask = xr.zeros_like(data.coords[combined_dim], dtype=bool)
    data = data.assign_coords(diverging=div_mask)
    data = dataset_to_dataarray(
        data, labeller=labeller, sample_dims=[combined_dim], label_type="vert"
    )
    if len(data.coords["label"].values) <= 1 or data.ndim != 2:
        raise ValueError(
            "Parallel plot requires at least two variables to plot. "
            "Please provide more than one variable."
        )

    # get labels and x-values
    x_labels = data.coords["label"].values
    x_values = np.arange(len(x_labels))

    new_sample_dims = [combined_dim, "label"]
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if plot_collection is None:
        pc_kwargs.setdefault("cols", None)
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, data)
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        colors = plot_bknd.get_default_aes("color", 1, {})
        pc_kwargs["aes"].setdefault("color", ["diverging"])
        pc_kwargs["aes"].setdefault("alpha", ["diverging"])
        pc_kwargs["color"] = pc_kwargs.get("color", [colors[0], "black"])
        pc_kwargs["alpha"] = pc_kwargs.get("alpha", [0.05, 0.1])
        plot_collection = PlotCollection.wrap(
            data.to_dataset(),
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals.setdefault("line", plot_collection.aes_set)

    # plot lines
    line_kwargs = copy(visuals.get("line", {}))
    if line_kwargs is not False:
        _, _, line_ignore = filter_aes(plot_collection, aes_by_visuals, "line", new_sample_dims)
        plot_collection.map(
            multiple_lines,
            "line",
            data=data,
            x_dim="label",
            xvalues=x_values,
            ignore_aes=line_ignore,
            **line_kwargs,
        )

    # x-axis label
    xticks_kwargs = copy(visuals.get("xticks", {}))
    if xticks_kwargs is not False:
        _, _, xticks_ignore = filter_aes(plot_collection, aes_by_visuals, "xticks", new_sample_dims)
        plot_collection.map(
            set_xticks,
            "xticks",
            labels=x_labels,
            values=x_values,
            rotation=0,
            ignore_aes=xticks_ignore,
            artist_dims={"labels": len(x_labels)},
            **xticks_kwargs,
        )

    return plot_collection
