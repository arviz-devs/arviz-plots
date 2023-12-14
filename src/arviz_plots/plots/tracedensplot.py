"""TraceDens plot code."""
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.utils import _var_names

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes
from arviz_plots.visuals import labelled_x, labelled_y, remove_ticks, line, line_xy, ecdf_line


def plot_trace_dens(
    dt,
    var_names=None,
    filter_vars=None,
    sample_dims=None,
    compact=True,
    kind=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    dens_kwargs=None,
    trace_kwargs=None,
    pc_kwargs=None,
):
    """Plot 1D marginal densities on the first row and iteration versus sampled
    values on the second.

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
    compact: bool, optional
        Plot multidimensional variables in a single plot. Defaults to True
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Defaults to only mapping properties to the density representation.
    trace_kwargs : mapping, optional
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
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if dens_kwargs is None:
        dens_kwargs = {}
    if trace_kwargs is None:
        trace_kwargs = {}
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

    aux_dim_list = [dim for dim in posterior.dims if dim not in {"chain", "draw"}]

    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("cols", ["__column__"])

        if compact:
            pc_kwargs.setdefault("rows", ["__variable__"])
        else:
            pc_kwargs.setdefault("rows", ["__variable__"]+aux_dim_list)

        plot_collection = PlotCollection.grid(
           posterior.expand_dims(__column__=2),
            backend=backend,
            **pc_kwargs,
        )

    if aes_map is None:
        aes_map = {"trace": plot_collection.aes_set}

    if labeller is None:
        labeller = BaseLabeller()

    density_dims, _, density_ignore = filter_aes(plot_collection, aes_map, "trace", sample_dims)
    
    if compact:
        density_dims += aux_dim_list


    # dens
    if kind == "kde":
        density = posterior.azstats.kde(dims=density_dims, **dens_kwargs.get("density", {}))
        plot_collection.map(
            line_xy, "dens", data=density, ignore_aes=density_ignore, coords={"__column__": 0}, **trace_kwargs.get("dens", {})
        )

    elif kind == "ecdf":
        density = posterior.azstats.ecdf(dims=density_dims, **dens_kwargs.get("density", {}))
        plot_collection.map(
            ecdf_line, "dens", data=density, ignore_aes=density_ignore, coords={"__column__": 0}, **trace_kwargs.get("dens", {}),
        )

    # trace
    plot_collection.map(
        line, "trace", data=posterior, ignore_aes=density_ignore,  coords={"__column__": 1}, **trace_kwargs.get("trace", {})
    )

    # aesthetics
    if kind == "kde":
        _, _, yticks_dens_ignore = filter_aes(plot_collection, aes_map, "yticks_dens", sample_dims)
        yticks_dens_kwargs = dens_kwargs.get("yticks_dens", {}).copy()

        plot_collection.map(
            remove_ticks,
            "yticks_dens",
            ignore_aes=yticks_dens_ignore,
            coords={"__column__": 0},
            **yticks_dens_kwargs,
        )

    _, xlabel_dens_aes, xlabel_dens_ignore = filter_aes(plot_collection, aes_map, "xlabel_dens", sample_dims)
    xlabel_dens_kwargs = dens_kwargs.get("xlabel_dens", {}).copy()

    if "color" not in xlabel_dens_aes:
        xlabel_dens_kwargs.setdefault("color", "black")
        #xlabel_dens_kwargs.setdefault("fontsize", "5")
    
    plot_collection.map(
        labelled_x,
        "xlabel_dens",
        ignore_aes=xlabel_dens_ignore,
        coords={"__column__": 0},
        subset_info=True,
        labeller=labeller,
        **xlabel_dens_kwargs,
    )

    _, xlabel_trace_aes, xlabel_trace_ignore = filter_aes(plot_collection, aes_map, "xlabel_trace", sample_dims)
    xlabel_trace_kwargs = dens_kwargs.get("xlabel_trace", {}).copy()

    if "color" not in xlabel_trace_aes:
        xlabel_trace_kwargs.setdefault("color", "black")
        #xlabel_trace_kwargs.setdefault("fontsize", "5")
    
    plot_collection.map(
        labelled_x,
        "xlabel_trace",
        ignore_aes=xlabel_trace_ignore,
        coords={"__column__": 1},
        subset_info=True,
        labeller=labeller,
        text="Steps",
        **xlabel_trace_kwargs,
    )

    _, ylabel_trace_aes, ylabel_trace_ignore = filter_aes(plot_collection, aes_map, "ylabel_trace", sample_dims)
    ylabel_trace_kwargs = trace_kwargs.get("ylabel_trace", {}).copy()

    if "color" not in ylabel_trace_aes:
        ylabel_trace_kwargs.setdefault("color", "black")
        #ylabel_trace_kwargs.setdefault("fontsize", "5")
    
    plot_collection.map(
        labelled_y,
        "xlabel_dens",
        ignore_aes=ylabel_trace_ignore,
        coords={"__column__": 1},
        subset_info=True,
        labeller=labeller,
        **ylabel_trace_kwargs,
    )



    return plot_collection
