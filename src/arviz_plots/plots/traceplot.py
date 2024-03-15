"""Trace plot code."""
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.utils import _var_names

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes
from arviz_plots.visuals import labelled_title, line


def plot_trace(
    dt,
    var_names=None,
    filter_vars=None,
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
        when plotted. Defaults to only mapping properties to the density representation.
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
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    var_names = _var_names(var_names, dt.posterior.ds, filter_vars)

    if var_names is None:
        posterior = dt.posterior.ds
    else:
        posterior = dt.posterior.ds[var_names]

    pc_kwargs.setdefault("aes", {"color": ["chain"]})

    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols", ["__variable__"] + [dim for dim in posterior.dims if dim not in sample_dims]
        )
        plot_collection = PlotCollection.wrap(
            posterior,
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {"trace": plot_collection.aes_set}
    if labeller is None:
        labeller = BaseLabeller()

    # trace
    _, _, trace_ignore = filter_aes(plot_collection, aes_map, "trace", sample_dims)
    plot_collection.map(
        line, "trace", data=posterior, ignore_aes=trace_ignore, **plot_kwargs.get("trace", {})
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
