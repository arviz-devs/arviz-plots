"""TraceDens plot code."""
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.utils import _var_names

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, get_size_of_var, scale_fig_size
from arviz_plots.visuals import (
    ecdf_line,
    labelled_x,
    labelled_y,
    line,
    line_xy,
    remove_ticks,
    ticks_size,
)


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
    plot_kwargs=None,
    pc_kwargs=None,
):
    """Plot 1D marginal densities and iteration versus sampled values.

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
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if dens_kwargs is None:
        dens_kwargs = {}
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

    aux_dim_list = [dim for dim in posterior.dims if dim not in sample_dims]

    figsize, textsize, linewidth = scale_fig_size(
        pc_kwargs.get("plot_grid_kws", {}).get("figsize", None),
        rows=get_size_of_var(posterior, compact=compact),
        cols=2,
    )

    plot_kwargs.setdefault("dens", {}).setdefault("lw", linewidth)
    plot_kwargs.setdefault("trace", {}).setdefault("lw", linewidth)

    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("cols", ["__column__"])
        pc_kwargs.setdefault("plot_grid_kws", {"figsize": figsize})

        if compact:
            pc_kwargs.setdefault("rows", ["__variable__"])
        else:
            pc_kwargs.setdefault("rows", ["__variable__"] + aux_dim_list)

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
            line_xy,
            "dens",
            data=density,
            ignore_aes=density_ignore,
            coords={"__column__": 0},
            **plot_kwargs.get("dens", {}),
        )

    elif kind == "ecdf":
        density = posterior.azstats.ecdf(dims=density_dims, **dens_kwargs.get("density", {}))
        plot_collection.map(
            ecdf_line,
            "dens",
            data=density,
            ignore_aes=density_ignore,
            coords={"__column__": 0},
            **plot_kwargs.get("dens", {}),
        )

    # trace
    plot_collection.map(
        line,
        "trace",
        data=posterior,
        ignore_aes=density_ignore,
        coords={"__column__": 1},
        **plot_kwargs.get("trace", {}),
    )

    ## aesthetics
    # Remove yticks, only for KDEs
    if kind == "kde":
        _, _, yticks_dens_ignore = filter_aes(plot_collection, aes_map, "yticks_dens", sample_dims)

        plot_collection.map(
            remove_ticks,
            "yticks_dens",
            ignore_aes=yticks_dens_ignore,
            coords={"__column__": 0},
        )

    # Add varnames as x and y labels
    _, labels_dens_aes, labels_dens_ignore = filter_aes(
        plot_collection, aes_map, "labels_dens", sample_dims
    )
    labels_dens_kwargs = dens_kwargs.get("labels_dens", {}).copy()

    if "color" not in labels_dens_aes:
        labels_dens_kwargs.setdefault("color", "black")

    labels_dens_kwargs.setdefault("fontsize", textsize)

    plot_collection.map(
        labelled_x,
        "labels_dens",
        ignore_aes=labels_dens_ignore,
        coords={"__column__": 0},
        subset_info=True,
        labeller=labeller,
        **labels_dens_kwargs,
    )

    plot_collection.map(
        labelled_y,
        "labels_dens",
        ignore_aes=labels_dens_ignore,
        coords={"__column__": 1},
        subset_info=True,
        labeller=labeller,
        **labels_dens_kwargs,
    )

    # Adjust ticks size
    plot_collection.map(
        ticks_size,
        "labels_dens",
        ignore_aes=labels_dens_ignore,
        subset_info=True,
        value=textsize,
        **labels_dens_kwargs,
    )

    # Add "Steps" as x_label for trace
    _, xlabel_trace_aes, xlabel_trace_ignore = filter_aes(
        plot_collection, aes_map, "xlabel_trace", sample_dims
    )
    xlabel_plot_kwargs = dens_kwargs.get("xlabel_trace", {}).copy()

    if "color" not in xlabel_trace_aes:
        xlabel_plot_kwargs.setdefault("color", "black")
        xlabel_plot_kwargs.setdefault("fontsize", textsize)

    plot_collection.map(
        labelled_x,
        "xlabel_trace",
        ignore_aes=xlabel_trace_ignore,
        coords={"__column__": 1},
        subset_info=True,
        labeller=labeller,
        text="Steps",
        **xlabel_plot_kwargs,
    )

    return plot_collection
