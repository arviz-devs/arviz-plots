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
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    process_group_variables_coords,
    set_grid_layout,
)
from arviz_plots.visuals import (
    ecdf_line,
    hist,
    label_plot,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_couple,
)


def plot_pair(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    plot_matrix=None,
    backend=None,
    marginal=True,
    marginal_kind="kde",
    marginal_stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "scatter",
            "divergence",
            "marginal",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "scatter",
            "divergence",
            "marginal",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    triangle="both",
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
        * marginal -> :func:`~.visuals.line_xy` or :func:`~.visuals.hist`
        * remove_axis -> optional argument passed to all above mentioned visuals plotting functions,
        can take value from  {"both", "x", "y"}. Defaults to ``False`` to skip removing axis.
    triangle : {"both", "upper", "lower"}, Defaults to "both"
        Which triangle of the pair plot to plot.
    **pc_kwargs
        Passed to :class:`arviz_plots.PlotMatrix`


    Returns
    -------
    PlotMatrix

    Examples
    --------
    Default plot_pair

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

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
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
    axis_to_remove = visuals.get("remove_axis", False)
    if marginal:
        axis_to_remove = False
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
        pc_kwargs["figure_kwargs"].setdefault("sharey", True)
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)
        if marginal:
            pc_kwargs["figure_kwargs"]["sharey"] = False
        plot_matrix = PlotMatrix(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    # scatter

    aes_by_visuals["scatter"] = {"overlay"}.union(aes_by_visuals.get("scatter", {}))
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

        if axis_to_remove:
            scatter_kwargs.setdefault("axis_to_remove", axis_to_remove)

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
        if marginal_stats is None:
            marginal_stats = {}
        else:
            marginal_stats = marginal_stats.copy()
        marginal_kwargs = copy(visuals.get("dist", {}))
        marginal_dims, marginal_aes, marginal_ignore = filter_aes(
            plot_matrix, aes_by_visuals, "marginal", sample_dims
        )
        if axis_to_remove:
            marginal_kwargs.setdefault("axis_to_remove", axis_to_remove)
        default_color = plot_bknd.get_default_aes("color", 1, {})[0]
        if "color" not in marginal_aes:
            marginal_kwargs.setdefault("color", default_color)

        if marginal_kind == "kde":
            density = distribution.azstats.kde(dim=marginal_dims, **marginal_stats.get("dist", {}))
            plot_matrix.map(
                line_xy, "marginal", data=density, ignore_aes=marginal_ignore, **marginal_kwargs
            )
        elif marginal_kind == "hist":
            hist_kwargs = marginal_stats.pop("dist", {}).copy()
            hist_kwargs.setdefault("density", True)
            density = distribution.azstats.histogram(dim=marginal_dims, **hist_kwargs)
            plot_matrix.map(
                hist,
                "marginal",
                data=density,
                ignore_aes=marginal_ignore,
                **marginal_kwargs,
            )
        elif marginal_kind == "ecdf":
            density = distribution.azstats.ecdf(dim=marginal_dims, **marginal_stats.get("dist", {}))
            plot_matrix.map(
                ecdf_line, "marginal", data=density, ignore_aes=marginal_ignore, **marginal_kwargs
            )

    # diagonal label of plots
    else:
        label_kwargs = copy(visuals.get("label", {}))

        if label_kwargs is not False:
            if axis_to_remove:
                label_kwargs.setdefault("axis_to_remove", axis_to_remove)
            _, _, label_ignore = filter_aes(plot_matrix, aes_by_visuals, "label", sample_dims)
            plot_matrix.map(
                label_plot,
                "label",
                subset_info=True,
                ignore_aes=label_ignore,
                labeller=labeller,
                **label_kwargs,
            )

    # divergence

    aes_by_visuals["divergence"] = {"overlay"}.union(aes_by_visuals.get("divergence", {}))
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
        if axis_to_remove:
            div_kwargs.setdefault("axis_to_remove", axis_to_remove)

        plot_matrix.map_triangle(
            scatter_couple,
            "divergence",
            triangle=triangle,
            data=distribution,
            ignore_aes=div_ignore,
            mask=divergence_mask,
            **div_kwargs,
        )

    # bottom plots xlabel
    if marginal and triangle in {"both", "lower"}:
        total = len(plot_matrix.viz.col_index.values)
        last_row = plot_matrix.viz.var_name_x[total - 1]
        remove_list = ["col_index", "row_index", "var_name_x", "var_name_y"]
        last_row_sel = {
            key[:-2]: value.item()
            for key, value in last_row.coords.items()
            if key not in remove_list and value.item() is not None
        }
        last_row_var_name = last_row.values.item()
        xlabel_kwargs = copy(visuals.get("xlabel", {}))
        if xlabel_kwargs is not False:
            _, _, xlabel_ignore = filter_aes(plot_matrix, aes_by_visuals, "xlabel", sample_dims)
            plot_matrix.map_row_col(
                labelled_x,
                var_name=last_row_var_name,
                selection=last_row_sel,
                orientation="row",
                data=distribution,
                ignore_aes=xlabel_ignore,
                labeller=labeller,
                subset_info=True,
                **xlabel_kwargs,
            )

        # left most plots ylabel
        first_col = plot_matrix.viz.var_name_y[0]
        first_col_sel = {
            key[:-2]: value.item()
            for key, value in first_col.coords.items()
            if key not in remove_list and value.item() is not None
        }
        first_col_var_name = first_col.values.item()
        ylabel_kwargs = copy(visuals.get("ylabel", {}))
        if ylabel_kwargs is not False:
            _, _, ylabel_ignore = filter_aes(plot_matrix, aes_by_visuals, "ylabel", sample_dims)
            plot_matrix.map_row_col(
                labelled_y,
                var_name=first_col_var_name,
                selection=first_col_sel,
                orientation="col",
                data=distribution,
                ignore_aes=ylabel_ignore,
                labeller=labeller,
                subset_info=True,
                **ylabel_kwargs,
            )
    elif marginal and triangle == "upper":
        diag_xlabel_kwargs = copy(visuals.get("diag_xlabel", {}))

        if diag_xlabel_kwargs is not False:
            if axis_to_remove:
                diag_xlabel_kwargs.setdefault("axis_to_remove", axis_to_remove)
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

    return plot_matrix
