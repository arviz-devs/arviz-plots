"""PsenseDist plot code."""
# pylint: disable=too-many-positional-arguments
from copy import copy
from importlib import import_module

from arviz_base import extract, rcParams
from arviz_stats.psense import _get_power_scale_weights
from xarray import concat

from arviz_plots.plot_collection import PlotCollection, process_facet_dims
from arviz_plots.plots.distplot import plot_dist
from arviz_plots.plots.utils import process_group_variables_coords


def plot_psense_dist(
    dt,
    alphas=None,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot power scaled posteriors.

    Parameters
    ----------
    dt : DataTree
        Input data
    alphas : tuple of float
        Lower and upper alpha values for power scaling. Defaults to (0.8, 1.25).
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal distribution.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection` when
        plotted.
        Valid keys are the same as for `plot_kwargs`.
    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * One of "kde", "ecdf", "dot" or "hist", matching the `kind` argument.

          * "kde" -> :func:`~.visuals.line_xy`
          * "ecdf" -> :func:`~.visuals.ecdf_line`


    stats_kwargs : mapping, optional
        Valid keys are:

        * density -> passed to kde, ecdf, ...

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotColletion`

    Returns
    -------
    PlotCollection
    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if stats_kwargs is None:
        stats_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        figsize = pc_kwargs.get("plot_grid_kws", {}).get("figsize", None)
        figsize_units = pc_kwargs.get("plot_grid_kws", {}).get("figsize_units", "inches")
        aux_dim_list = [dim for dim in distribution.dims if dim not in sample_dims]
        pc_kwargs.setdefault("rows", ["__variable__"] + aux_dim_list)
        aux_dim_list = [dim for dim in pc_kwargs["rows"] if dim != "__variable__"]
        row_dims = pc_kwargs["rows"]
    else:
        figsize, figsize_units = plot_bknd.get_figsize(plot_collection)
        aux_dim_list = list(
            set(
                dim for child in plot_collection.viz.children.values() for dim in child["plot"].dims
            ).difference({"column"})
        )
        row_dims = ["__variable__"] + aux_dim_list

    figsize = plot_bknd.scale_fig_size(
        figsize,
        rows=process_facet_dims(distribution, row_dims)[0],
        cols=2,
        figsize_units=figsize_units,
    )

    color_cycle = pc_kwargs.get("color", plot_bknd.get_default_aes("color", 3, {}))
    if len(color_cycle) <= 2:
        raise ValueError(
            f"Not enough values provided for color cycle, got {color_cycle} "
            "but at least 3 are needed"
        )
    import numpy as np

    pc_kwargs.setdefault("color", ["k"] + color_cycle)
    pc_kwargs.setdefault("y", np.linspace(-0.4, -0.05, 3))  # XXX can we use relative values?
    pc_kwargs.setdefault("aes", {"color": ["chain"], "y": ["chain"]})

    if plot_collection is None:
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        if "figsize" not in pc_kwargs["plot_grid_kws"]:
            pc_kwargs["plot_grid_kws"]["figsize"] = figsize
            pc_kwargs["plot_grid_kws"]["figsize_units"] = "dots"

        pc_kwargs["plot_grid_kws"].setdefault("sharex", "row")
        pc_kwargs["plot_grid_kws"].setdefault("sharey", "row")

        plot_collection = PlotCollection.grid(
            distribution.expand_dims(column=2).assign_coords(column=["prior", "likelihood"]),
            backend=backend,
            **pc_kwargs,
        )

    if kind == "hist":
        # XXX probably better to use a "step" histogram
        plot_kwargs.setdefault("hist", {"alpha": 0.3})
        # Also we should use the same number of "bins", but compute it from data
        stats_kwargs.setdefault("density", {"bins": 50, "density": True})

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    aes_map.setdefault("point_estimate", ["color", "y"])
    aes_map.setdefault("credible_interval", ["color", "y"])

    if alphas is None:
        alphas = (0.8, 1.25)

    # Here we are generating new datasets for the prior and likelihood
    # by resampling the original dataset with the power scale weights
    # Instead we should have weighted KDEs/ecdfs/etc
    dt_prior = new_dt(dt, "log_prior", alphas)
    dt_likelihood = new_dt(dt, "log_likelihood", alphas)

    # dens
    plot_kwargs_dist = {key: False for key in ("point_estimate_text", "text")}
    dist_kwargs = copy(plot_kwargs.get(kind, {}))
    plot_kwargs_dist[kind] = dist_kwargs

    plot_collection.coords = {"column": "prior"}
    plot_dist(
        dt_prior,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        kind=kind,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_map=aes_map,
        plot_kwargs=plot_kwargs_dist,
        stats_kwargs=stats_kwargs,
    )
    plot_collection.coords = None

    plot_collection.coords = {"column": "likelihood"}
    plot_dist(
        dt_likelihood,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        kind=kind,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_map=aes_map,
        plot_kwargs=plot_kwargs_dist,
        stats_kwargs=stats_kwargs,
    )
    plot_collection.coords = None

    return plot_collection


def new_dt(dt, group, alphas):
    """Replace Me."""
    resampled = []
    lower_w, upper_w = _get_power_scale_weights(dt, alphas, group=group)
    lower_w = lower_w.values.flatten()
    upper_w = upper_w.values.flatten()
    s_size = len(lower_w)

    for weights in (None, lower_w, upper_w):
        resampled.append(
            extract(
                dt, group="posterior", num_samples=s_size, weights=weights, random_seed=42
            ).drop("chain")
        )

    return concat(resampled, dim="chain").rename({"sample": "draw"})
