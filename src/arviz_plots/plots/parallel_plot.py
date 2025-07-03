"""Pair focus plot code."""
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import dataset_to_dataarray, extract, rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, get_group, set_wrap_layout
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
            "divergence",
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
    """Plot a fixed variable against other variables in the dataset.

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
        * divergence -> passed to :func:`~.visuals.multiple_lines`.
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
        >>> plot_pair_focus(
        >>>     dt,
        >>>     coords={"school": ["Choate","Deerfield"]},
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

    if coords is None:
        coords = {}

    data = extract(
        dt,
        group=group,
        var_names=var_names,
        sample_dims=sample_dims,
        filter_vars=filter_vars,
        keep_dataset=True,
    )
    data = data.sel(coords)
    data = dataset_to_dataarray(data, labeller=labeller, sample_dims=["sample"], label_type="vert")
    if len(data.coords["label"].values) <= 1 or data.ndim != 2:
        raise ValueError(
            "Parallel plot requires at least two variables to plot. "
            "Please provide more than one variable."
        )

    # get labels and x-values
    x_labels = data.coords["label"].values
    x_values = np.arange(len(x_labels))

    # create divergence mask
    div_kwargs = copy(visuals.get("divergence", {}))
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
        stacked_mask = divergence_mask.stack(sample=sample_dims)
        aligned_mask = stacked_mask.sel(sample=data.coords["sample"])
    else:
        aligned_mask = xr.DataArray(
            data=np.zeros(data.sizes["sample"], dtype=bool),
            dims=["sample"],
            coords={
                coord: data.coords[coord]
                for coord in data.coords
                if "sample" in data.coords[coord].dims
            },
            name="diverging",
        )

    new_sample_dims = ["sample", "label"]
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if plot_collection is None:
        pc_kwargs.setdefault("cols", None)
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, data)
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
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
    aes_by_visuals.setdefault("divergence", plot_collection.aes_set)

    line_kwargs = copy(visuals.get("line", {}))
    if line_kwargs is not False:
        _, line_aes, line_ignore = filter_aes(
            plot_collection, aes_by_visuals, "line", new_sample_dims
        )
        if "color" not in line_aes:
            colors = plot_bknd.get_default_aes("color", 1, {})
            line_kwargs.setdefault("color", colors[0])
        if "alpha" not in line_aes:
            line_kwargs.setdefault("alpha", 0.05)
        non_divergence_data = data.where(~aligned_mask)
        plot_collection.map(
            multiple_lines,
            "line",
            data=non_divergence_data,
            xvalues=x_values,
            overlay_dim="sample",
            ignore_aes=line_ignore,
            artist_dims={"sample": non_divergence_data.sizes["sample"]},
            **line_kwargs,
        )

    # divergence
    if aligned_mask.any().item():
        divergent_data = data.where(aligned_mask)
        _, div_aes, div_ignore = filter_aes(
            plot_collection, aes_by_visuals, "divergence", new_sample_dims
        )
        if "color" not in div_aes:
            div_kwargs.setdefault("color", "black")
        if "alpha" not in div_aes:
            div_kwargs.setdefault("alpha", 0.1)
        plot_collection.map(
            multiple_lines,
            "divergence",
            data=divergent_data,
            xvalues=x_values,
            overlay_dim="sample",
            ignore_aes=div_ignore,
            artist_dims={"sample": divergent_data.sizes["sample"]},
            **div_kwargs,
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
