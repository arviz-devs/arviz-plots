"""PsenseDist plot code."""
from importlib import import_module

from arviz_base import extract, rcParams
from arviz_base.labels import BaseLabeller
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
    # group="posterior",
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
    sample_dims = list(sample_dims)
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if stats_kwargs is None:
        stats_kwargs = {}
    else:
        stats_kwargs = stats_kwargs.copy()
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if alphas is None:
        alphas = (0.8, 1.25)

    # Here we are generating new datasets for the prior and likelihood
    # by resampling the original dataset with the power scale weights
    # Instead we could have weighted KDEs/ecdfs/etc
    ds_prior = new_ds(dt, "prior", alphas, sample_dims=sample_dims)
    ds_likelihood = new_ds(dt, "likelihood", alphas, sample_dims=sample_dims)
    distribution = concat([ds_prior, ds_likelihood], dim="group").assign_coords(
        {"group": ["prior", "likelihood"]}
    )
    distribution = process_group_variables_coords(
        distribution, group=None, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    if len(sample_dims) > 1:
        # sample dims will have been stacked and renamed by `new_ds`
        sample_dims = ["sample"]

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    color_cycle = pc_kwargs.get("color", plot_bknd.get_default_aes("color", 2, {}))
    if len(color_cycle) < 2:
        raise ValueError(
            f"Not enough values provided for color cycle, got {color_cycle} "
            "but at least 2 are needed"
        )

    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["plot_grid_kws"].setdefault("sharex", "row")
        pc_kwargs["plot_grid_kws"].setdefault("sharey", "row")

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("color", [color_cycle[0], "black", color_cycle[1]])
        pc_kwargs.setdefault("y", [-0.4, -0.225, -0.05])  # XXX can we use relative values?
        pc_kwargs["aes"].setdefault("color", ["alpha"])
        pc_kwargs["aes"].setdefault("y", ["alpha"])
        pc_kwargs.setdefault("cols", ["group"])
        pc_kwargs.setdefault(
            "rows",
            ["__variable__"]
            + [dim for dim in distribution.dims if dim not in sample_dims + ["group", "alpha"]],
        )

        figsize = pc_kwargs["plot_grid_kws"].get("figsize", None)
        figsize_units = pc_kwargs["plot_grid_kws"].get("figsize_units", "inches")
        row_dims = pc_kwargs["rows"]
        if figsize is None:
            figsize = plot_bknd.scale_fig_size(
                figsize,
                rows=process_facet_dims(distribution, row_dims)[0],
                cols=2,
                figsize_units=figsize_units,
            )
            figsize_units = "dots"
        pc_kwargs["plot_grid_kws"]["figsize"] = figsize
        pc_kwargs["plot_grid_kws"]["figsize_units"] = figsize_units

        plot_collection = PlotCollection.grid(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    plot_kwargs.setdefault("point_estimate_text", False)

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

    plot_dist(
        distribution,
        var_names=None,
        filter_vars=None,
        group=None,
        coords=None,
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

    return plot_collection


def new_ds(dt, group, alphas, sample_dims):
    """Resample the dataset with the power scale weights."""
    lower_w, upper_w = _get_power_scale_weights(dt, alphas, group=group, sample_dims=sample_dims)
    lower_w = lower_w.values.flatten()
    upper_w = upper_w.values.flatten()
    s_size = len(lower_w)

    idxs_to_drop = sample_dims if len(sample_dims) == 1 else ["sample"] + sample_dims
    idxs_to_drop = set(idxs_to_drop).union(
        [
            idx
            for idx in dt["posterior"].xindexes
            if any(dim in dt["posterior"][idx].dims for dim in sample_dims)
        ]
    )
    resampled = [
        extract(
            dt,
            group="posterior",
            sample_dims=sample_dims,
            num_samples=s_size,
            weights=weights,
            random_seed=42,
            resampling_method="stratified",
        ).drop_indexes(idxs_to_drop)
        for weights in (lower_w, upper_w)
    ]
    resampled.insert(
        1, extract(dt, group="posterior", sample_dims=sample_dims).drop_indexes(idxs_to_drop)
    )

    return concat(resampled, dim="alpha").assign_coords(alpha=[alphas[0], 1, alphas[1]])
