"""Pair plot code."""
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams, xarray_sel_iter
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_matrix import PlotMatrix
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    process_group_variables_coords,
    set_grid_layout,
)
from arviz_plots.visuals import (
    label_plot,
    labelled_x,
    labelled_y,
    remove_matrix_axis,
    scatter_couple,
    set_ticklabel_visibility,
)


def plot_pair(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    marginal=True,
    marginal_kind=None,
    triangle="lower",
    plot_matrix=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "scatter",
            "divergence",
            "dist",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "label",
            "xlabel",
            "ylabel",
            "diag_xlabel",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "scatter",
            "divergence",
            "dist",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[
        Literal[
            "dist",
            "credible_interval",
            "point_estimate",
        ],
        Mapping[str, Any] | xr.Dataset,
    ] = None,
    **pc_kwargs,
):
    """Plot all variables against each other in the dataset.

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
    marginal : bool, default True
        Whether to plot marginal distributions on the diagonal.
    marginal_kind : {"kde", "hist", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    triangle : {"both", "upper", "lower"}, Defaults to "both"
        Which triangle of the pair plot to plot.
    plot_matrix : PlotMatrix, optional
    backend : {"matplotlib", "bokeh", "plotly", "none"}, optional
        Plotting backend to use. Defaults to ``rcParams["plot.backend"]``
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_matrix`
        when plotted. Valid keys are the same as for `visuals`.
        By default, there are no aesthetic mappings at all
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * scatter -> passed to :func:`~.visuals.scatter_couple`
        * divergence -> passed to :func:`~.visuals.scatter_couple`. Defaults to False.
        * dist -> depending on the value of `marginal_kind` passed to:

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * credible_interval -> passed to :func:`~arviz_plots.visuals.line_x`
        * point_estimate -> passed to :func:`~arviz_plots.visuals.scatter_x`
        * point_estimate_text -> passed to :func:`~arviz_plots.visuals.point_estimate_text`
        * label -> passed to :func:`~.visuals.label_plot`, applicable only if `marginal` is False
        * xlabel -> passed to :func:`~.visuals.labelled_x`, applicable only if `marginal` is True
        * ylabel -> passed to :func:`~.visuals.labelled_y`, applicable only if `marginal` is True
        * diag_xlabel -> passed to :func:`~.visuals.labelled_x` for diag-plots if `marginal` is True

    stats : mapping, optional
        Valid keys are:

        * dist -> passed to kde, ecdf, ...
        * credible_interval -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotMatrix`


    Returns
    -------
    PlotMatrix

    Examples
    --------
    plot_pair with divergence and marginal at diagonal (`triangle` is set to "lower" by default).

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>>     visuals={"divergence": True},
        >>>     marginal=True,
        >>>     marginal_kind="hist",
        >>> )

    plot_pair with marginal at diagonal and without divergence.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>>     marginal=True,
        >>>     marginal_kind="hist",
        >>> )

    plot_pair with `triangle` set to "upper" (In this case the xlabels are on the diagonal plots).

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>>     visuals={"divergence": True},
        >>>     marginal=True,
        >>>     marginal_kind="hist",
        >>>     triangle="upper",
        >>> )

    plot_pair with `triangle` set to "both" (x/y labels are on bottom/left most plots).

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>>     visuals={"divergence": True},
        >>>     marginal=True,
        >>>     marginal_kind="hist",
        >>>     triangle="both",
        >>> )

    plot_pair without marginal ( in this case labels of variables take place of marginals).

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>>     visuals={"divergence": True},
        >>>     marginal=False,
        >>> )


    plot_pair without marginal and `triangle` set to "upper".

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>>     visuals={"divergence": True},
        >>>     marginal=False,
        >>>     triangle="upper",
        >>> )

    plot_pair without marginal and `triangle` set to "both".

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair(
        >>>     dt,
        >>>     var_names=["mu", "tau"],
        >>>     visuals={"divergence": True},
        >>>     marginal=False,
        >>>     triangle="both",
        >>> )


    .. minigallery:: plot_pair

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

    if labeller is None:
        labeller = BaseLabeller()
    if backend is None:
        if plot_matrix is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_matrix.backend

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if plot_matrix is None:
        pc_kwargs.setdefault(
            "facet_dims",
            ["__variable__"] + [dim for dim in distribution.dims if dim not in sample_dims],
        )
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pairs = tuple(
            xarray_sel_iter(
                distribution, skip_dims={dim for dim in distribution.dims if dim in sample_dims}
            )
        )
        n_pairs = len(pairs)
        pc_kwargs = set_grid_layout(
            pc_kwargs, plot_bknd, distribution, num_rows=n_pairs, num_cols=n_pairs
        )
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        pc_kwargs["figure_kwargs"].setdefault("sharex", "col")
        if not marginal:
            pc_kwargs["figure_kwargs"].setdefault("sharey", "row")

        plot_matrix = PlotMatrix(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals["scatter"] = {"overlay"}.union(
        aes_by_visuals.get("scatter", plot_matrix.aes_set)
    )
    aes_by_visuals["divergence"] = {"overlay"}.union(aes_by_visuals.get("divergence", {}))
    aes_by_visuals["dist"] = aes_by_visuals.get("dist", plot_matrix.aes_set.difference({"overlay"}))
    aes_by_visuals["credible_interval"] = aes_by_visuals.get("credible_interval", {})
    aes_by_visuals["point_estimate"] = aes_by_visuals.get("point_estimate", {})
    aes_by_visuals["point_estimate_text"] = aes_by_visuals.get("point_estimate_text", {})

    # scatter
    scatter_kwargs = copy(visuals.get("scatter", {}))
    if scatter_kwargs is not False:
        _, scatter_aes, scatter_ignore = filter_aes(
            plot_matrix, aes_by_visuals, "scatter", sample_dims
        )

        if "color" not in scatter_aes:
            colors = plot_bknd.get_default_aes("color", 1, {})
            scatter_kwargs.setdefault("color", colors[0])

        if "width" not in scatter_aes:
            scatter_kwargs.setdefault("width", 0)

        if "alpha" not in scatter_aes:
            scatter_kwargs.setdefault("alpha", 0.5)

        plot_matrix.map_triangle(
            scatter_couple,
            "scatter",
            triangle=triangle,
            data=distribution,
            ignore_aes=scatter_ignore,
            **scatter_kwargs,
        )

    # marginal
    if marginal is not False:
        if stats is None:
            stats = {}
        else:
            stats = stats.copy()

        dist_plot_visuals = {}
        dist_plot_aes_by_visuals = {}
        dist_plot_stats = {}
        marginal_dist_kwargs = copy(visuals.get("dist", {}))
        marginal_ci_kwargs = copy(visuals.get("credible_interval", False))
        marginal_point_estimate_kwargs = copy(visuals.get("point_estimate", False))
        marginal_point_estimate_text_kwargs = copy(visuals.get("point_estimate_text", False))

        dist_plot_visuals["dist"] = marginal_dist_kwargs
        dist_plot_visuals["credible_interval"] = marginal_ci_kwargs
        dist_plot_visuals["point_estimate"] = marginal_point_estimate_kwargs
        dist_plot_visuals["point_estimate_text"] = marginal_point_estimate_text_kwargs
        dist_plot_visuals["title"] = False
        dist_plot_visuals["remove_axis"] = False
        dist_plot_visuals["rug"] = False

        dist_plot_aes_by_visuals["dist"] = aes_by_visuals.get(
            "dist", plot_matrix.aes_set.difference({"overlay"})
        )
        dist_plot_aes_by_visuals["credible_interval"] = aes_by_visuals.get("credible_interval", {})
        dist_plot_aes_by_visuals["point_estimate"] = aes_by_visuals.get("point_estimate", {})
        dist_plot_aes_by_visuals["point_estimate_text"] = aes_by_visuals.get(
            "point_estimate_text", {}
        )

        dist_plot_stats["dist"] = stats.get("dist", {})
        dist_plot_stats["credible_interval"] = stats.get("credible_interval", {})
        dist_plot_stats["point_estimate"] = stats.get("point_estimate", {})

        plot_matrix = plot_dist(
            distribution,
            sample_dims=sample_dims,
            kind=marginal_kind,
            plot_collection=plot_matrix,
            backend=backend,
            labeller=labeller,
            aes_by_visuals=dist_plot_aes_by_visuals,
            visuals=dist_plot_visuals,
            stats=dist_plot_stats,
        )

    # diagonal labels of rows and cols
    else:
        label_kwargs = copy(visuals.get("label", {}))
        if label_kwargs is not False:
            lim_low = distribution.min(dim=sample_dims)
            lim_high = distribution.max(dim=sample_dims)
            text_center = (lim_high + lim_low) / 2
            _, _, label_ignore = filter_aes(plot_matrix, aes_by_visuals, "label", sample_dims)
            plot_matrix.map(
                label_plot,
                "label",
                subset_info=True,
                labeller=labeller,
                x=text_center,
                y=text_center,
                lim_low=lim_low,
                lim_high=lim_high,
                ignore_aes=label_ignore,
                **label_kwargs,
            )

    # divergence
    div_kwargs = copy(visuals.get("divergence", False))
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
        _, div_aes, div_ignore = filter_aes(plot_matrix, aes_by_visuals, "divergence", sample_dims)
        if "color" not in div_aes:
            div_kwargs.setdefault("color", "black")
        if "alpha" not in div_aes:
            div_kwargs.setdefault("alpha", 0.5)

        plot_matrix.map_triangle(
            scatter_couple,
            "divergence",
            triangle=triangle,
            data=distribution,
            ignore_aes=div_ignore,
            mask=divergence_mask,
            **div_kwargs,
        )

    # bottom plots xlabel and left plots ylabel
    if marginal and triangle in {"both", "lower"}:
        xlabel_kwargs = copy(visuals.get("xlabel", {}))
        if xlabel_kwargs is not False:
            _, _, xlabel_ignore = filter_aes(plot_matrix, aes_by_visuals, "xlabel", sample_dims)
            plot_matrix.map_row(
                labelled_x,
                "xlabel",
                index=-1,
                data=distribution,
                ignore_aes=xlabel_ignore,
                labeller=labeller,
                subset_info=True,
                **xlabel_kwargs,
            )

        ylabel_kwargs = copy(visuals.get("ylabel", {}))
        if ylabel_kwargs is not False:
            _, _, ylabel_ignore = filter_aes(plot_matrix, aes_by_visuals, "ylabel", sample_dims)
            plot_matrix.map_col(
                labelled_y,
                "ylabel",
                index=0,
                data=distribution,
                ignore_aes=ylabel_ignore,
                labeller=labeller,
                subset_info=True,
                **ylabel_kwargs,
            )
    elif marginal and triangle == "upper":
        diag_xlabel_kwargs = copy(visuals.get("diag_xlabel", {}))

        if diag_xlabel_kwargs is not False:
            _, _, diag_xlabel_ignore = filter_aes(
                plot_matrix, aes_by_visuals, "diag_xlabel", sample_dims
            )
            plot_matrix.map(
                labelled_x,
                "diag_xlabel",
                subset_info=True,
                ignore_aes=diag_xlabel_ignore,
                labeller=labeller,
                **diag_xlabel_kwargs,
            )

    # set ticklabel visibility
    set_ticklabel_visibility_kwargs = {}
    if triangle == "upper":
        _, _, set_ticklabel_visibility_ignore = filter_aes(
            plot_matrix, aes_by_visuals, "set_ticklabel_visibility", sample_dims
        )
        plot_matrix.map(
            set_ticklabel_visibility,
            "set_ticklabel_visibility",
            axis="x",
            visible=True,
            ignore_aes=set_ticklabel_visibility_ignore,
            **set_ticklabel_visibility_kwargs,
        )

        if not marginal:
            plot_matrix.map(
                set_ticklabel_visibility,
                "set_ticklabel_visibility",
                axis="y",
                visible=True,
                ignore_aes=set_ticklabel_visibility_ignore,
                **set_ticklabel_visibility_kwargs,
            )

    # default removal of axis for better visualization
    _, _, default_remove_axis_ignore = filter_aes(
        plot_matrix, aes_by_visuals, "default_remove_axis", sample_dims
    )
    # if triangle="upper" then remove the lower triangle axes
    if triangle == "upper":
        plot_matrix.map_triangle(
            remove_matrix_axis,
            "default_remove_axis",
            triangle="lower",
            axis="both",
            ignore_aes=default_remove_axis_ignore,
        )
    # if triangle="lower" then remove the upper triangle axes
    elif triangle == "lower":
        plot_matrix.map_triangle(
            remove_matrix_axis,
            "default_remove_axis",
            triangle="upper",
            axis="both",
            ignore_aes=default_remove_axis_ignore,
        )
    return plot_matrix
