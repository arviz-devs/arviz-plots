"""Pair focus plot code."""
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import dataset_to_dataarray, rcParams
from arviz_base.labels import BaseLabeller
from scipy.stats import rankdata

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import multiple_lines, set_xticks


def normalize(da: xr.DataArray, norm_method: str, norm_dim: str) -> xr.DataArray:
    """Apply normalization to an xarray DataArray along the given dimension."""
    if norm_method is None:
        return da

    if norm_method == "normal":
        mean = da.mean(dim=norm_dim)
        std_dev = da.std(dim=norm_dim)
        return (da - mean) / std_dev

    if norm_method == "minmax":
        min_val = da.min(dim=norm_dim)
        max_val = da.max(dim=norm_dim)
        return (da - min_val) / (max_val - min_val)

    if norm_method == "rank":
        return xr.apply_ufunc(
            rankdata,
            da,
            input_core_dims=[[norm_dim]],
            output_core_dims=[[norm_dim]],
            kwargs={"method": "average", "axis": -1},
            dask="parallelized",
            output_dtypes=[float],
        )

    raise ValueError(f"{norm_method} is not supported. Use normal, minmax or rank.")


def plot_parallel(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    norm_method=None,
    label_type="flat",
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
    norm_method : Method for normalizing the data.
        Methods include normal, minmax and rank. Defaults to None
    label_type : {"flat", "vert"}, default "flat"
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
        * xticks -> passed to :func:`~.visuals.set_xticks`. Defaults to False.


    **pc_kwargs
        Passed to :meth:`arviz_plots.PlotCollection.wrap`


    Returns
    -------
    PlotCollection

    Examples
    --------
    Default plot_parallel without normalization and with default ``label_type`` as "flat".

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_parallel, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_parallel(
        >>>     dt,
        >>>     var_names=["theta", "tau", "mu"],
        >>> )

    parallel_plot with ``norm_method`` set to "normal" and ``label_type`` set to "vert"
    and ``xticks`` rotation set to 30 degrees.

    .. plot::
        :context: close-figs

        >>> plot_parallel(
        >>>     dt,
        >>>     var_names=["theta", "tau", "mu"],
        >>>     norm_method="normal",
        >>>     label_type="vert",
        >>> )

    parallel_plot with ``norm_method`` set to "minmax".

    .. plot::
        :context: close-figs

        >>> plot_parallel(
        >>>     dt,
        >>>     var_names=["theta", "tau", "mu"],
        >>>     norm_method="minmax",
        >>> )


    .. minigallery:: plot_parallel

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if visuals is None:
        visuals = {}

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
        data, labeller=labeller, sample_dims=[combined_dim], label_type=label_type
    )
    if len(data.coords["label"].values) <= 1 or data.ndim != 2:
        raise ValueError(
            "Parallel plot requires at least two variables to plot. "
            "Please provide more than one variable."
        )

    # get labels and x-values
    x_labels = data.coords["label"].values
    x_values = np.arange(len(x_labels))

    # normalize data
    data = normalize(data, norm_method, combined_dim)

    new_sample_dims = [combined_dim, "label"]
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if plot_collection is None:
        pc_kwargs.setdefault("cols", None)
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, data)
        colors = plot_bknd.get_default_aes("color", 1, {})
        pc_kwargs["aes"].setdefault("color", ["diverging"])
        pc_kwargs["aes"].setdefault("alpha", ["diverging"])
        pc_kwargs["color"] = pc_kwargs.get("color", ["black", colors[0]])
        pc_kwargs["alpha"] = pc_kwargs.get("alpha", [0.03, 0.2])
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
        xticks_kwargs["rotation"] = xticks_kwargs.get("rotation", 90)
        plot_collection.map(
            set_xticks,
            "xticks",
            labels=x_labels,
            values=x_values,
            ignore_aes=xticks_ignore,
            artist_dims={"labels": len(x_labels)},
            **xticks_kwargs,
        )

    return plot_collection
