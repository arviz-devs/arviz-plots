"""PsenseDist plot code."""
# pylint: disable=too-many-positional-arguments
from copy import copy
from importlib import import_module

from arviz_base import extract, rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.psense import _get_power_scale_weights
from xarray import concat

from arviz_plots.plot_collection import PlotCollection, process_facet_dims
from arviz_plots.plots.distplot import plot_dist
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import labelled_title


def plot_psense_dist(
    dt,
    alphas=None,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind=None,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
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
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to rcParam :data:`stats.point_estimate`
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to ``rcParams["stats.ci_prob"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * One of "kde", "ecdf", "dot" or "hist", matching the `kind` argument.

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * credible_interval -> passed to :func:`~arviz_plots.visuals.line_x`
        * point_estimate -> passed to :func:`~arviz_plots.visuals.scatter_x`
        * point_estimate_text -> passed to :func:`~arviz_plots.visuals.point_estimate_text`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats_kwargs : mapping, optional
        Valid keys are:

        * density -> passed to kde, ecdf, ...
        * credible_interval -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.wrap`

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

    plot_kwargs.setdefault("point_estimate_text", False)

    # Middle chain is the reference chain (alpha == 1)
    pc_kwargs.setdefault("color", [color_cycle[0], "k", color_cycle[1]])
    pc_kwargs.setdefault("y", [-0.4, -0.225, -0.05])  # XXX can we use relative values?
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

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    aes_map.setdefault("point_estimate", ["color", "y"])
    aes_map.setdefault("credible_interval", ["color", "y"])

    if labeller is None:
        labeller = BaseLabeller()

    if kind == "hist":
        # Histograms are not great for overlapping distributions
        # But "step" histograms may be slightly easier to interpret than bars histograms
        # Using the same number of "bins" should help too
        plot_kwargs.setdefault("hist", {})
        plot_kwargs["hist"].setdefault("alpha", 0.3)
        plot_kwargs["hist"].setdefault("edgecolor", None)
        stats_kwargs.setdefault("density", {"density": True})

    if alphas is None:
        alphas = (0.8, 1.25)

    # Here we are generating new datasets for the prior and likelihood
    # by resampling the original dataset with the power scale weights
    # Instead we could have weighted KDEs/ecdfs/etc
    ds_prior = new_ds(dt, "log_prior", alphas)
    ds_likelihood = new_ds(dt, "log_likelihood", alphas)

    plot_collection.coords = {"column": "prior"}
    plot_dist(
        ds_prior,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        kind=kind,
        point_estimate=point_estimate,
        ci_kind=ci_kind,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_map=aes_map,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
    )
    plot_collection.coords = None

    plot_collection.coords = {"column": "likelihood"}
    plot_dist(
        ds_likelihood,
        var_names=var_names,
        filter_vars=filter_vars,
        group=group,
        coords=coords,
        sample_dims=sample_dims,
        kind=kind,
        point_estimate=point_estimate,
        ci_kind=ci_kind,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_map=aes_map,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
    )
    plot_collection.coords = None

    # Overwrite the title from plot_dist to include the prior and likelihood labels
    title_kwargs = copy(plot_kwargs.get("title", {}))
    if title_kwargs is not False:
        _, title_aes, title_ignore = filter_aes(plot_collection, aes_map, "title", sample_dims)
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


def new_ds(dt, group, alphas):
    """Resample the dataset with the power scale weights."""
    lower_w, upper_w = _get_power_scale_weights(dt, alphas, group=group)
    lower_w = lower_w.values.flatten()
    upper_w = upper_w.values.flatten()
    s_size = len(lower_w)

    resampled = [extract(dt, group="posterior").drop("chain")]
    for weights in (lower_w, upper_w):
        resampled.append(
            extract(
                dt,
                group="posterior",
                num_samples=s_size,
                weights=weights,
                random_seed=42,
                resampling_method="stratified",
            ).drop("chain")
        )

    # Middle chain is the reference chain (alpha == 1)
    return concat(resampled, dim="chain").rename({"sample": "draw"}).sel(chain=[1, 0, 2])
