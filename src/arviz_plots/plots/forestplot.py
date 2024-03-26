"""Forest plot code."""
from importlib import import_module

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import annotate_label, fill_between_y, line_x, remove_axis, scatter_x


def plot_forest(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    combined=False,
    point_estimate=None,
    ci_kind=None,
    ci_probs=None,
    labels=None,
    shade_label=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot 1D marginal credible intervals in a single plot.

    For a general introduction to batteries included functions like this one and common
    usage examples see :ref:`plots_intro`

    Parameters
    ----------
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.

        ``plot_forest`` uses the dimension "column" (creating it if necessary) to generate the grid
        then adds the intervals+point estimates to its "forest" coordinate
        and labels to its "labels" coordinates. The data used to plot is then the subset
        ``column="forest"``.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    combined : bool, False
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to ``rcParams["plot.point_estimate"]``
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_probs : (float, float), optional
        Indicates the probabilities that should be contained within the plotted credible intervals.
        It should be sorted as the elements refer to the probabilities of the "trunk" and "twig"
        elements. Defaults to ``(0.5, rcParams["stats.ci_prob"])``
    labels : iterable of str, optional
        Iterable with the dimensions to be labelled in the plot. By default all dimensions.
        It can include the special "__variable__" indicator, and does so by default.
    shade_label : str, default None
        Element of `labels` that should be used to add shading horizontal strips to the plot.
        Note that labels and credible intervals are plotted in different :term:`plots`.
        The shading is applied to both plots, and the spacing between them is set to 0
        *if possible*, which is not always the case (one notable example being matplotlib's
        constrained layout).
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : iterable of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs` except "ticklabels"
        which doesn't apply and "twig" and "trunk" which, similarly to `stats_kwargs`
        take the same aesthetics through the "credible_interval" key.

        By default, aesthetic mappings are generated for: y, alpha, overlay and color
        (if multiple models are present). All aesthetic mappings but alpha are applied
        to both the credible intervals and the point estimate; overlay is applied
        to labels; and both overlay and alpha are applied to the shade.

        "overlay" is a dummy aesthetic to trigger looping over variables and/or
        dimensions using all aesthetics in every iteration. "alpha" gets two
        values (0, 0.3) in order to trigger the alternate shading effect.
    plot_kwargs : mapping, optional
        Valid keys are:

        * trunk, twig -> passed to :func:`~.visuals.line_x`
        * point_estimate -> passed to :func:`~.visuals.scatter_x`
        * labels -> passed to :func:`~.visuals.annotate_label`
        * shade -> passed to :func:`~.visuals.fill_between_y`
        * ticklabels -> passed to :func:`~.backend.xticks`

    stats_kwargs : mapping, optional
        Valid keys are:

        * credible_interval -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Notes
    -----
    The separation between variables and all its coordinate values is set to 1.
    The only two exceptions to this are the dimensions named "chain" and "model"
    in case they are present, which get a smaller spacing to give a sense of
    grouping among visual elements that only differ on their chain or model id.

    Examples
    --------
    Default forest plot for a single model:

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_forest, style
        >>> style.use("arviz-clean")
        >>> from arviz_base import load_arviz_data
        >>> centered = load_arviz_data('centered_eight')
        >>> non_centered = load_arviz_data('non_centered_eight')
        >>> pc = plot_forest(centered)

    Default forest plot for multiple models:

    .. plot::
        :context: close-figs

        >>> pc = plot_forest({"centered": centered, "non centered": non_centered})
        >>> pc.add_legend("model")

    Single model forest plot with color mapped to the variable (mapping which is also applied
    to the labels) and alternate shading per school.
    Moreover, to ensure the shading looks continuous, we'll specify we don't want to use
    constrained layout (set by the "arviz-clean" theme) and to avoid having the labels
    too squished we'll set the ``width_ratios`` for
    :func:`~arviz_plots.backend.create_plotting_grid` via ``pc_kwargs``.

    .. plot::
        :context: close-figs

        >>> pc = plot_forest(
        >>>     non_centered,
        >>>     var_names=["theta", "mu", "theta_t", "tau"],
        >>>     pc_kwargs={
        >>>         "aes": {"color": ["__variable__"]},
        >>>         "plot_grid_kws": {"width_ratios": [1, 2], "layout": "none"}
        >>>     },
        >>>     aes_map={"labels": ["color"]},
        >>>     shade_label="school",
        >>> )

    Extend the forest plot with an extra :term:`plot` with ess estimates.
    To achieve that, we manually add a "column" dimension with size 3.
    ``plot_forest`` only plots on the "labels" and "forest" coordinate values,
    leaving the "ess" coordinate empty. Afterwards, we manually use
    :meth:`.PlotCollection.map` with the ess result as data on the "ess" column
    to plot their values.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import visuals
        >>> from arviz_stats.base import ess
        >>>
        >>> c_aux = centered["posterior"].expand_dims(
        >>>     column=3
        >>> ).assign_coords(column=["labels", "forest", "ess"])
        >>> pc = plot_forest(c_aux, combined=True)
        >>> pc.map(
        >>>     visuals.scatter_x, "ess", data=ess(centered),
        >>>     coords={"column": "ess"}, color="C0"
        >>> )

    Note that we are using the same :class:`~.PlotCollection`, so when using
    ``map`` all the same aesthetic mappings used by ``plot_forest`` are used.

    """
    if ci_kind not in ("hdi", "eti", None):
        raise ValueError("ci_kind must be either 'hdi' or 'eti'")

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if ci_probs is None:
        rc_ci_prob = rcParams["stats.ci_prob"]
        ci_probs = (0.5, rc_ci_prob) if rc_ci_prob > 0.5 else (0.5 * rc_ci_prob, rc_ci_prob)
    if ci_probs[0] > ci_probs[1]:
        raise ValueError("First element of ci_probs must be smaller than the second")
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"] if "stats.ci_kind" in rcParams else "eti"
    if point_estimate is None:
        point_estimate = rcParams["plot.point_estimate"]
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if stats_kwargs is None:
        stats_kwargs = {}

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    labellable_dims = ["__variable__"] + [
        dim for dim in distribution.dims if (dim not in {"model", "column"}.union(sample_dims))
    ]
    if labels is None:
        labels = labellable_dims

    if shade_label is not None and shade_label not in labels:
        raise ValueError("shade_label must be one of the elements in labels argument")

    if plot_collection is None:
        pc_data = distribution
        if "column" not in pc_data:
            pc_data = pc_data.expand_dims(column=2).assign_coords(column=["labels", "forest"])
        elif ("forest" not in pc_data.column) or ("labels" not in pc_data.column):
            raise ValueError(
                "Found colum dimension in input data but required coordinates "
                "'labels' and 'forest' are missing."
            )
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["plot_grid_kws"].setdefault("sharey", True)
        width_ratios = xr.ones_like(pc_data.column, dtype=float)
        width_ratios.loc[{"column": "forest"}] = 3 if len(labels) < 3 else 2
        pc_kwargs["plot_grid_kws"].setdefault("width_ratios", width_ratios.values)
        if shade_label is not None:
            pc_kwargs["plot_grid_kws"].setdefault("plot_hspace", 0)
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("y", labellable_dims)
        pc_kwargs["aes"].setdefault("overlay", labellable_dims)
        pc_kwargs["aes"].setdefault("alpha", [shade_label])
        pc_kwargs.setdefault("alpha", [0, 0.3])
        if "model" in distribution.dims:
            pc_kwargs["aes"].setdefault("color", ["model"])
        plot_collection = PlotCollection.grid(
            pc_data,
            backend=backend,
            **pc_kwargs,
        )
    if plot_collection.aes is None:
        plot_collection.generate_aes_dt()

    if "column" in distribution.dims:
        distribution = distribution.sel(column="forest")

    # fine tune y position for model and chain
    y_ds = plot_collection.get_aes_as_dataset("y")
    add_factor = 0
    shift = 0
    if combined and "model" in distribution.dims:
        add_factor = 0.2
        shift = xr.DataArray(
            np.linspace(-0.2, 0.2, distribution.sizes["model"]),
            coords={"model": distribution.model},
        )
    elif ("model" in distribution.dims) and ("chain" in distribution.dims):
        add_factor = 0.2
        model_spacing = xr.DataArray(
            np.linspace(-0.2, 0.2, distribution.sizes["model"]),
            coords={"model": distribution.model},
        )
        chain_lim = 0.4 * (model_spacing[1] - model_spacing[0])
        chain_spacing = xr.DataArray(
            np.linspace(-chain_lim, chain_lim, distribution.sizes["chain"]),
            coords={"chain": distribution.chain},
        )
        shift = model_spacing + chain_spacing
    elif "chain" in distribution.dims and not combined:
        add_factor = 0.2
        shift = xr.DataArray(
            np.linspace(-0.2, 0.2, distribution.sizes["chain"]),
            coords={"chain": distribution.chain},
        )
    y_ds = y_ds.max().to_array().max() - add_factor - y_ds - shift
    plot_collection.update_aes_from_dataset("y", y_ds)

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("credible_interval", plot_collection.aes_set.difference({"alpha"}))
    aes_map.setdefault("point_estimate", plot_collection.aes_set.difference({"alpha"}))
    aes_map["labels"] = {"overlay"}.union(aes_map.get("labels", {}))
    aes_map["shade"] = {"overlay", "alpha"}.union(aes_map.get("shade", {}))
    if labeller is None:
        labeller = BaseLabeller()

    # compute credible interval
    ci_dims, ci_aes, ci_ignore = filter_aes(
        plot_collection, aes_map, "credible_interval", sample_dims
    )
    if ci_kind == "eti":
        ci_trunk = distribution.azstats.eti(
            prob=ci_probs[0], dims=ci_dims, **stats_kwargs.get("credible_interval", {})
        )
        ci_twig = distribution.azstats.eti(
            prob=ci_probs[1], dims=ci_dims, **stats_kwargs.get("credible_interval", {})
        )
    elif ci_kind == "hdi":
        ci_trunk = distribution.azstats.hdi(
            prob=ci_probs[0], dims=ci_dims, **stats_kwargs.get("credible_interval", {})
        )
        ci_twig = distribution.azstats.hdi(
            prob=ci_probs[1], dims=ci_dims, **stats_kwargs.get("credible_interval", {})
        )

    # add labels and shading first, so forest plot is rendered on top
    _, lab_aes, lab_ignore = filter_aes(plot_collection, aes_map, "labels", sample_dims)
    cumulative_label = []
    x = 0
    for label in labellable_dims:
        cumulative_label.append(label)
        # each variable+coord combination has the space between i.5 and (i+1).5 "reserved"
        # if there multiple models/combined all lines are plotted in the central region
        # of .4 width, if single model+combined, the line is at the center
        # shade extend is the value to add to the max/substract to the min to shade the
        # whole unit regions reserved to variables/coords
        shade_extend = 0.5 if add_factor == 0 else 0.3
        if label not in labels:
            continue
        lab_kwargs = plot_kwargs.get("labels", {}).copy()
        if "color" not in lab_aes:
            lab_kwargs.setdefault("color", "black")
        if x == 0:
            lab_kwargs.setdefault("horizontal_align", "left")
        if x == len(labels) - 1:
            lab_kwargs.setdefault("horizontal_align", "right")
        if label == "__variable__":
            y_max = y_ds.max() + shade_extend
            y_min = y_ds.min() - shade_extend
        else:
            reduce_dims = [dim for dim in y_ds.dims if dim not in cumulative_label]
            y_max = y_ds.max(reduce_dims) + shade_extend
            y_min = y_ds.min(reduce_dims) - shade_extend
        y = (y_max + y_min) / 2
        if shade_label == label:
            _, shade_aes, shade_ignore = filter_aes(plot_collection, aes_map, "shade", sample_dims)
            shade_kwargs = plot_kwargs.get("shade", {}).copy()
            if "color" not in shade_aes:
                shade_kwargs.setdefault("color", "gray")
            shade_data = xr.concat((y_min, y_max), "kwarg").assign_coords(
                kwarg=["y_bottom", "y_top"]
            )
            shade_start = -0.1 if x == 0 else x - 0.6
            xlim_labels = [-0.1, len(labels) - 0.9]
            plot_collection.map(
                fill_between_y,
                "shade",
                data=shade_data,
                x=[shade_start, xlim_labels[1]],
                coords={"column": "labels"},
                ignore_aes=shade_ignore,
                **shade_kwargs,
            )
            ci_global_min = ci_twig.min().to_array().min().item()
            ci_global_max = ci_twig.max().to_array().max().item()
            ci_range_extend = 0.1 * (ci_global_max - ci_global_min)
            xlim_forest = [ci_global_min - ci_range_extend, ci_global_max + ci_range_extend]
            plot_collection.map(
                fill_between_y,
                "shade",
                data=shade_data,
                x=xlim_forest,
                coords={"column": "forest"},
                ignore_aes=shade_ignore,
                **shade_kwargs,
            )
        plot_collection.map(
            annotate_label,
            f"{label.strip('_')}_label",
            data=y,
            x=x,
            dim=None if label == "__variable__" else label,
            subset_info=True,
            coords={"column": "labels"},
            ignore_aes=lab_ignore,
            **lab_kwargs,
        )
        x += 1
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    plot_bknd.xticks(
        np.arange(len(labels)),
        [label.strip("_") for label in labels],
        plot_collection.get_target(None, {"column": "labels"}),
        **plot_kwargs.get("ticklabels", {}),
    )

    # plot credible interval
    default_color = plot_bknd.get_default_aes("color", 1, {})[0]
    twig_kwargs = plot_kwargs.get("twig", {}).copy()
    twig_kwargs.setdefault("width", 0.7)
    if "color" not in ci_aes:
        twig_kwargs.setdefault("color", default_color)
    plot_collection.map(
        line_x,
        "twig",
        data=ci_twig,
        ignore_aes=ci_ignore,
        coords={"column": "forest"},
        **twig_kwargs,
    )
    trunk_kwargs = plot_kwargs.get("trunk", {}).copy()
    trunk_kwargs.setdefault("width", 2)
    if "color" not in ci_aes:
        trunk_kwargs.setdefault("color", default_color)
    plot_collection.map(
        line_x,
        "trunk",
        data=ci_trunk,
        ignore_aes=ci_ignore,
        coords={"column": "forest"},
        **trunk_kwargs,
    )

    # point estimate
    pe_dims, pe_aes, pe_ignore = filter_aes(plot_collection, aes_map, "point_estimate", sample_dims)
    if point_estimate == "median":
        point = distribution.median(dim=pe_dims, **stats_kwargs.get("point_estimate", {}))
    elif point_estimate == "mean":
        point = distribution.mean(dim=pe_dims, **stats_kwargs.get("point_estimate", {}))
    else:
        raise NotImplementedError("coming soon")

    pe_kwargs = plot_kwargs.get("point_estimate", {}).copy()
    if "color" not in pe_aes:
        pe_kwargs.setdefault("color", default_color)
    if "facecolor" not in pe_aes:
        pe_kwargs.setdefault("facecolor", "white")
    plot_collection.map(
        scatter_x,
        "point_estimate",
        data=point,
        ignore_aes=pe_ignore,
        coords={"column": "forest"},
        **pe_kwargs,
    )
    if shade_label is not None:
        plot_bknd.xlim(xlim_labels, plot_collection.get_target(None, {"column": "labels"}))
        plot_bknd.xlim(xlim_forest, plot_collection.get_target(None, {"column": "forest"}))

    plot_collection.map(
        remove_axis, store_artist=False, axis="y", ignore_aes=plot_collection.aes_set
    )

    return plot_collection
