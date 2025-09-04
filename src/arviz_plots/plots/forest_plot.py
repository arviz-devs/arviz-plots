"""Forest plot code."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection, process_facet_dims
from arviz_plots.plots.utils import (
    filter_aes,
    get_contrast_colors,
    get_visual_kwargs,
    process_group_variables_coords,
)
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
    aes_by_visuals: Mapping[
        Literal[
            "credible_interval",
            "point_estimate",
            "labels",
            "shade",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "trunk",
            "twig",
            "point_estimate",
            "labels",
            "shade",
            "ticklabels",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[
        Literal["trunk", "twig", "point_estimate"], Mapping[str, Any] | xr.Dataset
    ] = None,
    **pc_kwargs,
):
    """Plot 1D marginal credible intervals in a single plot.

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
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    combined : bool, default False
        Whether to plot intervals for each chain or not. Ignored when the "chain" dimension
        is not present.
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to rcParam :data:`stats.point_estimate`
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_probs : (float, float), optional
        Indicates the probabilities that should be contained within the plotted credible intervals.
        It should be sorted as the elements refer to the probabilities of the "trunk" and "twig"
        elements. Defaults to ``(0.5, rcParams["stats.ci_prob"])``
    labels : sequence of str, optional
        Sequence with the dimensions to be labelled in the plot. By default all dimensions
        except "chain" and "model" (if present). The order of `labels` is ignored,
        only elements being present in it matters.
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
    aes_by_visuals : mapping of {str : sequence of str or False}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals` except "ticklabels"
        and "remove_axis" which do not apply, and "twig" and "trunk" which
        take the same aesthetics through the "credible_interval" key.

        By default, aesthetic mappings are generated for: y, alpha, overlay and color
        (if multiple models are present). All aesthetic mappings but alpha are applied
        to both the credible intervals and the point estimate; overlay is applied
        to labels; and both overlay and alpha are applied to the shade.

        "overlay" is a dummy aesthetic to trigger looping over variables and/or
        dimensions using all aesthetics in every iteration. "alpha" gets two
        values (0, 0.3) in order to trigger the alternate shading effect.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * trunk, twig -> passed to :func:`~.visuals.line_x`
        * point_estimate -> passed to :func:`~.visuals.scatter_x`
        * labels -> passed to :func:`~.visuals.annotate_label`
        * shade -> passed to :func:`~.visuals.fill_between_y`
        * ticklabels -> passed to :func:`~.backend.xticks`
        * remove_axis -> not passed anywhere, can only take ``False`` as value to skip calling
          :func:`~.visuals.remove_axis`

    stats : mapping, optional
        Valid keys are:

        * trunk, twig -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    **pc_kwargs
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

    See Also
    --------
    :ref:`plots_intro` :
        General introduction to batteries-included plotting functions, common use and logic overview
    plot_ridge : Visual representation of marginal distributions over the y axis

    Examples
    --------
    Single model forest plot with color mapped to the variable (mapping which is also applied
    to the labels) and alternate shading per school.
    Moreover, to ensure the shading looks continuous, we'll specify we don't want to use
    constrained layout (set by the "arviz-variat" theme) and to avoid having the labels
    too squished we'll set the ``width_ratios`` for
    :func:`~arviz_plots.backend.none.create_plotting_grid` via ``pc_kwargs``.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_forest, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> non_centered = load_arviz_data('non_centered_eight')
        >>> pc = plot_forest(
        >>>     non_centered,
        >>>     var_names=["theta", "mu", "theta_t", "tau"],
        >>>     aes={"color": ["__variable__"]},
        >>>     figure_kwargs={"width_ratios": [1, 2], "layout": "none"},
        >>>     aes_by_visuals={"labels": ["color"]},
        >>>     shade_label="school",
        >>> )

    .. minigallery:: plot_forest

    """
    if ci_kind not in ("hdi", "eti", None):
        raise ValueError("ci_kind must be either 'hdi' or 'eti'")

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if ci_probs is None:
        rc_ci_prob = rcParams["stats.ci_prob"]
        ci_probs = (0.5, rc_ci_prob) if rc_ci_prob > 0.5 else (0.5 * rc_ci_prob, rc_ci_prob)
    if ci_probs[0] > ci_probs[1]:
        raise ValueError("First element of ci_probs must be smaller than the second")
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"]
    if point_estimate is None:
        point_estimate = rcParams["stats.point_estimate"]
    if visuals is None:
        visuals = {}
    if stats is None:
        stats = {}

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    labellable_dims = ["__variable__"] + [
        dim for dim in distribution.dims if (dim not in {"model", "column"}.union(sample_dims))
    ]
    if labels is None:
        labels = labellable_dims
    if not combined and "chain" not in distribution.dims:
        combined = True

    labels_kwargs = get_visual_kwargs(visuals, "labels")
    if labels_kwargs is False:
        raise ValueError("visuals['labels'] can't be False, use labels=[] to remove all labels")
    shade_kwargs = get_visual_kwargs(visuals, "shade")
    if shade_kwargs is False:
        raise ValueError("visuals['shade'] can't be False, use shade_label=None to remove shading")

    if shade_label is not None and shade_label not in labels:
        raise ValueError("shade_label must be one of the elements in labels argument")

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    bg_color = plot_bknd.get_background_color()
    contrast_color, contrast_gray_color = get_contrast_colors(bg_color=bg_color, gray_flag=True)

    given_plotcollection = True
    if plot_collection is None:
        given_plotcollection = False
        pc_data = distribution
        if "column" not in pc_data:
            pc_data = pc_data.expand_dims(column=2).assign_coords(column=["labels", "forest"])
        elif ("forest" not in pc_data.column) or ("labels" not in pc_data.column):
            raise ValueError(
                "Found colum dimension in input data but required coordinates "
                "'labels' and 'forest' are missing."
            )
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharey", True)
        width_ratios = xr.ones_like(pc_data.column, dtype=float)
        width_ratios.loc[{"column": "forest"}] = 3 if len(labels) < 3 else 2
        pc_kwargs["figure_kwargs"].setdefault("width_ratios", width_ratios.values)
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        alpha_aes_dims = False
        if shade_label is not None:
            pc_kwargs["figure_kwargs"].setdefault("plot_hspace", 0)
            alpha_aes_dims = pc_kwargs["aes"].get("alpha", [shade_label])
            pc_kwargs["aes"]["alpha"] = alpha_aes_dims
        pc_kwargs["aes"].setdefault("y", labellable_dims)
        pc_kwargs["aes"].setdefault("overlay", labellable_dims)
        if alpha_aes_dims is not False:
            if ("__variable__" in pc_kwargs["aes"]["alpha"]) or all(
                shade_label in da.dims for da in distribution.values()
            ):
                pc_kwargs.setdefault("alpha", [0, 0.3])
            else:
                # trigger inclusion of neutral element in aes cycle
                pc_kwargs.setdefault("alpha", [0, 0, 0.3])
        if "model" in distribution.dims:
            pc_kwargs["aes"].setdefault("color", ["model"])
        figsize = pc_kwargs.get("figure_kwargs", {}).get("figsize", None)
        figsize_units = pc_kwargs.get("figure_kwargs", {}).get("figsize_units", "inches")
        if figsize is None:
            coeff = 0.2
            n_blocks = process_facet_dims(
                pc_data, [dim for dim in pc_kwargs["aes"]["y"] if dim not in ("chain", "model")]
            )[0]
            if not combined and "chain" in distribution.dims:
                coeff += 0.1
            if "model" in distribution.dims:
                coeff += 0.1 * distribution.sizes["model"]
            figsize = plot_bknd.scale_fig_size(
                figsize,
                rows=1 + coeff * n_blocks,
                cols=process_facet_dims(pc_data, pc_kwargs["cols"])[0],
                figsize_units=figsize_units,
            )
            figsize_units = "dots"
        pc_kwargs["figure_kwargs"]["figsize"] = figsize
        pc_kwargs["figure_kwargs"]["figsize_units"] = figsize_units
        plot_collection = PlotCollection.grid(
            pc_data,
            backend=backend,
            **pc_kwargs,
        )

    if "column" in distribution.dims:
        distribution = distribution.sel(column="forest")

    if combined:
        chain_mapped_to_aes = set(
            aes_key
            for var_name, child in plot_collection.aes.children.items()
            for aes_key, aes_vals in child.items()
            if "chain" in aes_vals.dims
        )
        if chain_mapped_to_aes:
            raise ValueError(
                f"Found properties {chain_mapped_to_aes} mapped to the chain dimension, "
                "but combined=True. Set combined=False or modify the aesthetic mappings"
            )

    # fine tune y position for model and chain
    add_factor = 0.2 if (not combined) or ("model" in distribution.dims) else 0
    y_ds = plot_collection.get_aes_as_dataset("y")
    if not given_plotcollection:
        shift = 0
        if combined and "model" in distribution.dims:
            shift = xr.DataArray(
                np.linspace(-0.2, 0.2, distribution.sizes["model"]),
                coords={"model": distribution.model},
            )
        elif (not combined) and ("model" in distribution.dims):
            model_spacing = xr.DataArray(
                np.linspace(-0.2, 0.2, distribution.sizes["model"]),
                coords={"model": distribution.model},
            )
            chain_lim = 0.4 * (model_spacing[1] - model_spacing[0]).item()
            chain_spacing = xr.DataArray(
                np.linspace(-chain_lim, chain_lim, distribution.sizes["chain"]),
                coords={"chain": distribution.chain},
            )
            shift = model_spacing + chain_spacing
        elif not combined:
            shift = xr.DataArray(
                np.linspace(-0.2, 0.2, distribution.sizes["chain"]),
                coords={"chain": distribution.chain},
            )
        y_ds = y_ds.max().to_array().max() - add_factor - y_ds - shift
        plot_collection.update_aes_from_dataset("y", y_ds)

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals.setdefault("credible_interval", plot_collection.aes_set.difference({"alpha"}))
    aes_by_visuals.setdefault("point_estimate", plot_collection.aes_set.difference({"alpha"}))
    aes_by_visuals["labels"] = {"overlay"}.union(aes_by_visuals.get("labels", {}))
    aes_by_visuals["shade"] = {"overlay", "alpha"}.union(aes_by_visuals.get("shade", {}))
    if labeller is None:
        labeller = BaseLabeller()

    # compute credible interval
    ci_dims, ci_aes, ci_ignore = filter_aes(
        plot_collection, aes_by_visuals, "credible_interval", sample_dims
    )
    twig_kwargs = get_visual_kwargs(visuals, "twig")
    trunk_kwargs = get_visual_kwargs(visuals, "trunk")
    if ci_kind == "eti":
        ci_fun = distribution.azstats.eti
    elif ci_kind == "hdi":
        ci_fun = distribution.azstats.hdi
    if twig_kwargs is not False:
        twig_stats = stats.get("twig", {})
        if isinstance(twig_stats, xr.Dataset):
            ci_twig = twig_stats
        else:
            ci_twig = ci_fun(prob=ci_probs[1], dim=ci_dims, **twig_stats)
    if trunk_kwargs is not False:
        trunk_stats = stats.get("trunk", {})
        if isinstance(trunk_stats, xr.Dataset):
            ci_trunk = trunk_stats
        else:
            ci_trunk = ci_fun(prob=ci_probs[0], dim=ci_dims, **trunk_stats)

    # compute point estimate
    pe_kwargs = get_visual_kwargs(visuals, "point_estimate")
    if pe_kwargs is not None:
        pe_dims, pe_aes, pe_ignore = filter_aes(
            plot_collection, aes_by_visuals, "point_estimate", sample_dims
        )
        pe_stats = stats.get("point_estimate", {})
        if isinstance(pe_stats, xr.Dataset):
            point = pe_stats
        elif point_estimate == "median":
            point = distribution.median(dim=pe_dims, **pe_stats)
        elif point_estimate == "mean":
            point = distribution.mean(dim=pe_dims, **pe_stats)
        elif point_estimate == "mode":
            point = distribution.azstats.mode(dim=pe_dims, **pe_stats)
        else:
            raise NotImplementedError(f"Point estimate '{point_estimate}' not implemented")

    if twig_kwargs is not False:
        x_range = ci_twig
    elif trunk_kwargs is not False:
        x_range = ci_trunk
    elif pe_kwargs is not False:
        x_range = point
    else:
        x_range = xr.ones_like(distribution)

    # add labels and shading first, so forest plot is rendered on top
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
        if label == "__variable__":
            y_max = y_ds.max() + shade_extend
            y_min = y_ds.min() - shade_extend
        else:
            reduce_dims = [dim for dim in y_ds.dims if dim not in cumulative_label]
            y_max = y_ds.max(reduce_dims) + shade_extend
            y_min = y_ds.min(reduce_dims) - shade_extend
        y = (y_max + y_min) / 2
        if shade_label == label:
            _, shade_aes, shade_ignore = filter_aes(
                plot_collection, aes_by_visuals, "shade", sample_dims
            )
            if "color" not in shade_aes:
                shade_kwargs.setdefault("color", contrast_gray_color)
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
            ci_global_min = x_range.min().to_array().min().item()
            ci_global_max = x_range.max().to_array().max().item()
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
        _, lab_aes, lab_ignore = filter_aes(plot_collection, aes_by_visuals, "labels", sample_dims)
        extra_ignore_aes = []
        for aes_key in lab_aes:
            if aes_key == "overlay":
                continue
            aes_ds = plot_collection.get_aes_as_dataset(aes_key)
            if set(aes_ds.dims).difference(cumulative_label):
                extra_ignore_aes.append(aes_key)
        lab_aes = set(lab_aes).difference(extra_ignore_aes)
        lab_ignore = set(lab_ignore).union(extra_ignore_aes)
        lab_kwargs = labels_kwargs.copy()
        if "color" not in lab_aes:
            lab_kwargs.setdefault("color", contrast_color)
        if x == 0:
            lab_kwargs.setdefault("horizontal_align", "left")
        if x == len(labels) - 1:
            lab_kwargs.setdefault("horizontal_align", "right")
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
    ticklabel_kwargs = get_visual_kwargs(visuals, "ticklabels")
    if ticklabel_kwargs is not False:
        plot_bknd.xticks(
            np.arange(len(labels)),
            [label.strip("_") for label in labellable_dims if label in labels],
            plot_collection.get_target(None, {"column": "labels"}),
            **ticklabel_kwargs,
        )

    # plot credible interval
    default_color = plot_bknd.get_default_aes("color", 1, {})[0]
    if twig_kwargs is not False:
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
    if trunk_kwargs is not False:
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
    if pe_kwargs is not False:
        if "color" not in pe_aes:
            pe_kwargs.setdefault("color", default_color)
        if "facecolor" not in pe_aes:
            pe_kwargs.setdefault("facecolor", "white")
        if "width" not in pe_aes:
            pe_kwargs.setdefault("width", 1)
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

    if visuals.get("remove_axis", True) is not False:
        plot_collection.map(
            remove_axis,
            store_artist=backend == "none",
            axis="y",
            ignore_aes=plot_collection.aes_set,
        )

    return plot_collection
