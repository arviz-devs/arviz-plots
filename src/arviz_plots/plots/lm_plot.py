"""lm plot code."""
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, List, Literal

import arviz_stats as azs
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_contrast_colors,
    get_group,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import fill_between_y, labelled_x, labelled_y, line_xy, scatter_xy


def plot_lm(
    dt,
    y=None,
    x=None,
    y_pred=None,
    x_pred=None,
    filter_vars=None,
    group="predictions",
    coords=None,
    sample_dims=None,
    ci_prob=None,
    ci_kind=None,
    line_kind="mean",
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "ci_line",
            "mean_line",
            "fill",
            "scatter",
            "xlabel",
            "ylabel",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "ci_line",
            "mean_line",
            "fill",
            "scatter",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[
        Literal["credible_interval",],
        Mapping[str, Any] | xr.Dataset,
    ] = None,
    **pc_kwargs,
):
    """Posterior predictive and mean plots for regression-like data..

    Parameters
    ----------
    dt : DataTree
        Input data

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

    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"] if "stats.ci_kind" in rcParams else "eti"

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    if stats is None:
        stats = {}
    else:
        stats = stats.copy()

    if labeller is None:
        labeller = BaseLabeller()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    obs_data = get_group(dt, "observed_data")
    if y is None:
        y = list(obs_data.data_vars)[0]

    if not isinstance(y, xr.DataArray | xr.Dataset):
        y = process_group_variables_coords(
            dt, group="observed_data", var_names=y, filter_vars=filter_vars, coords=coords
        )

    const_data = get_group(dt, "constant_data")
    if x is None:
        x = list(const_data.data_vars)

    if not isinstance(x, xr.DataArray | xr.Dataset):
        x = process_group_variables_coords(
            dt, group="constant_data", var_names=x, filter_vars=filter_vars, coords=coords
        )

    (target_var,) = y.data_vars
    independent_var = list(x.data_vars)

    if y_pred is None:
        y_pred = target_var

    if not isinstance(y_pred, xr.DataArray | xr.Dataset):
        y_pred = process_group_variables_coords(
            dt,
            group=group,
            var_names=y_pred,
            filter_vars=filter_vars,
            coords=coords,
        )

    if x_pred is None:
        x_pred = independent_var

    if not isinstance(x_pred, xr.DataArray | xr.Dataset):
        if group == "predictions":
            x_pred = process_group_variables_coords(
                dt,
                group="predictions_constant_data",
                var_names=x_pred,
                filter_vars=filter_vars,
                coords=coords,
            )
        else:
            x_pred = x

    if isinstance(ci_prob, List):
        x_with_prob = x.expand_dims(dim={"prob": ci_prob})
    else:
        x_with_prob = x

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    bg_color = plot_bknd.get_background_color()
    _, contrast_gray_color = get_contrast_colors(bg_color=bg_color, gray_flag=True)
    if plot_collection is None:
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if isinstance(ci_prob, List):
            alpha_dims = pc_kwargs["aes"].get("alpha", [])
            if "prob" not in alpha_dims:
                alpha_dims.append("prob")
            pc_kwargs["aes"]["alpha"] = alpha_dims

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, x)
        plot_collection = PlotCollection.wrap(
            x_with_prob,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals.setdefault("line", plot_collection.aes_set)
    if isinstance(ci_prob, List):
        aes_by_visuals.setdefault("ci_line", {"alpha"})
        aes_by_visuals.setdefault("fill", {"alpha"})

    # calculations for credible interval
    if ci_kind == "eti":
        if isinstance(ci_prob, List):
            ci_data = xr.concat(
                [
                    azs.eti(
                        y_pred, dim=sample_dims, prob=p, **stats.get("credible_interval", {})
                    ).expand_dims(prob=[p])
                    for p in ci_prob
                ],
                dim="prob",
            )
        else:
            ci_data = y_pred.azstats.eti(prob=ci_prob, **stats.get("credible_interval", {}))
    elif ci_kind == "hdi":
        if isinstance(ci_prob, List):
            ci_data = xr.concat(
                [
                    azs.hdi(
                        y_pred, dim=sample_dims, prob=p, **stats.get("credible_interval", {})
                    ).expand_dims(prob=[p])
                    for p in ci_prob
                ],
                dim="prob",
            )
        else:
            ci_data = y_pred.azstats.hdi(prob=ci_prob, **stats.get("credible_interval", {}))
    if line_kind == "mean":
        line_data = y_pred.mean(dim=["chain", "draw"])
    elif line_kind == "median":
        line_data = y_pred.median(dim=["chain", "draw"])
    colors = plot_bknd.get_default_aes("color", 2, {})
    lines = plot_bknd.get_default_aes("linestyle", 2, {})

    ci_lower = ci_data.sel(ci_bound="lower")
    ci_upper = ci_data.sel(ci_bound="upper")

    # upper and lower lines of credible interval
    ci_line_kwargs = copy(visuals.get("ci_line", False))
    if ci_line_kwargs is not False:
        _, ci_line_aes, ci_line_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ci_line", sample_dims
        )
        if "color" not in ci_line_aes:
            ci_line_kwargs.setdefault("color", contrast_gray_color)

        if "linestyle" not in ci_line_aes:
            ci_line_kwargs.setdefault("linestyle", lines[1])

        plot_collection.map(
            line_xy,
            "ci_line",
            x=x_pred,
            y=ci_lower[target_var],
            ignore_aes=ci_line_ignore,
            **ci_line_kwargs,
        )

        plot_collection.map(
            line_xy,
            "ci_line",
            x=x_pred,
            y=ci_upper[target_var],
            ignore_aes=ci_line_ignore,
            **ci_line_kwargs,
        )

    # fill between lines of credible interval
    fill_kwargs = copy(visuals.get("fill", {}))
    if fill_kwargs is not False:
        _, _, fill_ignore = filter_aes(plot_collection, aes_by_visuals, "fill", sample_dims)
        plot_collection.map(
            fill_between_y,
            "fill",
            x=x_pred,
            y_bottom=ci_lower[target_var],
            y_top=ci_upper[target_var],
            ignore_aes=fill_ignore,
            color=colors[1],
            **fill_kwargs,
        )

    # mean line
    mean_line_kwargs = copy(visuals.get("mean_line", {}))
    if mean_line_kwargs is not False:
        _, _, mean_line_ignore = filter_aes(
            plot_collection, aes_by_visuals, "mean_line", sample_dims
        )
        plot_collection.map(
            line_xy,
            "mean_line",
            x=x_pred,
            y=line_data[target_var],
            ignore_aes=mean_line_ignore,
            **mean_line_kwargs,
        )

    # scatter plot
    original_scatter_kwargs = copy(visuals.get("scatter", {}))
    if original_scatter_kwargs is not False:
        _, scatter_aes, scatter_ignore = filter_aes(
            plot_collection, aes_by_visuals, "scatter", sample_dims
        )
        if "alpha" not in scatter_aes:
            original_scatter_kwargs.setdefault("alpha", 0.3)
        if "color" not in scatter_aes:
            original_scatter_kwargs.setdefault("color", contrast_gray_color)
        if "width" not in scatter_aes:
            original_scatter_kwargs.setdefault("width", 0)
        plot_collection.map(
            scatter_xy,
            "scatter",
            x=x,
            y=y[target_var],
            ignore_aes=scatter_ignore,
            **original_scatter_kwargs,
        )

    # x-axis label
    xlabel_kwargs = copy(visuals.get("xlabel", {}))
    if xlabel_kwargs is not False:
        _, _, xlabel_ignore = filter_aes(plot_collection, aes_by_visuals, "xlabel", sample_dims)
        plot_collection.map(
            labelled_x,
            "xlabel",
            data=x,
            labeller=labeller,
            subset_info=True,
            ignore_aes=xlabel_ignore,
            **xlabel_kwargs,
        )

    # y-axis label
    ylabel_kwargs = copy(visuals.get("ylabel", {}))
    if ylabel_kwargs is not False:
        _, _, ylabel_ignore = filter_aes(plot_collection, aes_by_visuals, "ylabel", sample_dims)
        plot_collection.map(
            labelled_y,
            "ylabel",
            text=target_var,
            ignore_aes=ylabel_ignore,
            **ylabel_kwargs,
        )

    return plot_collection
