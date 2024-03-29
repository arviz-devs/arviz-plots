"""Trace plot code."""
import numpy as np
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, get_group, process_group_variables_coords
from arviz_plots.visuals import labelled_title, line, trace_rug


def plot_trace(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    pc_kwargs=None,
):
    """Plot iteration versus sampled values.

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
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Defaults to only mapping properties to the trace lines.
    plot_kwargs : mapping, optional
        Valid keys are:

        * trace -> passed to visuals.line
        * divergence -> passed to visuals.line_x

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection`

    Returns
    -------
    PlotCollection
    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if plot_collection is None:
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("color", ["chain"])
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in distribution.dims if dim not in sample_dims]
        )
        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("trace", plot_collection.aes_set)
    aes_map.setdefault("divergence", {"overlay"})

    if labeller is None:
        labeller = BaseLabeller()

    # trace
    trace_kwargs = plot_kwargs.get("trace", {}).copy()
    default_xname = sample_dims[0] if len(sample_dims) == 1 else "draw"
    if (default_xname not in distribution.dims) or (
        not np.issubdtype(distribution[default_xname].dtype, np.number)
    ):
        default_xname = None
    xname = trace_kwargs.get("xname", default_xname)
    trace_kwargs["xname"] = xname
    _, _, trace_ignore = filter_aes(plot_collection, aes_map, "trace", sample_dims)
    plot_collection.map(
        line,
        "trace",
        data=distribution,
        ignore_aes=trace_ignore,
        **trace_kwargs,
    )

    # divergences
    sample_stats = get_group(dt, "sample_stats", allow_missing=True)
    if (
        sample_stats is not None
        and "diverging" in sample_stats.data_vars
        and np.any(sample_stats.diverging)
    ):
        divergence_mask = dt.sample_stats.diverging
        _, div_aes, div_ignore = filter_aes(plot_collection, aes_map, "divergence", sample_dims)
        divergence_kwargs = plot_kwargs.get("divergence", {}).copy()
        if "color" not in div_aes:
            divergence_kwargs.setdefault("color", "black")
        if "marker" not in div_aes:
            divergence_kwargs.setdefault("marker", "|")
        if "width" not in div_aes:
            divergence_kwargs.setdefault("width", 1.5)
        if "size" not in div_aes:
            divergence_kwargs.setdefault("size", 30)

        plot_collection.map(
            trace_rug,
            "divergence",
            data=distribution,
            ignore_aes=div_ignore,
            xname=xname,
            y=distribution.min(sample_dims),
            mask=divergence_mask,
            **divergence_kwargs,
        )

    # aesthetics
    _, title_aes, title_ignore = filter_aes(plot_collection, aes_map, "title", sample_dims)
    title_kwargs = plot_kwargs.get("title", {}).copy()
    if "color" not in title_aes:
        title_kwargs.setdefault("color", "black")
    plot_collection.map(
        labelled_title,
        "title",
        ignore_aes=title_ignore,
        subset_info=True,
        labeller=labeller,
        **title_kwargs,
    )

    return plot_collection
