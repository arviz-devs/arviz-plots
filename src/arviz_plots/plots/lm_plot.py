"""lm plot code."""
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_contrast_colors,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import fill_between_y, labelled_x, labelled_y, line_xy, scatter_xy


def plot_lm(
    dt,
    target_data=None,
    independent_data=None,
    prediction_data=None,
    ci_mean_independent_data=None,
    filter_vars=None,
    prediction_group="predictions",
    coords=None,
    sample_dims=None,
    ci_prob=None,
    ci_kind=None,
    smooth=True,
    plot_collection=None,
    backend=None,
    aes_by_visuals: Mapping[
        Literal[
            "line",
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
            "line",
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

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend
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

    if target_data is None:
        target_data = process_group_variables_coords(
            dt, group="observed_data", var_names="y", filter_vars=filter_vars, coords=coords
        )
    elif isinstance(target_data, str):
        target_data = process_group_variables_coords(
            dt, group="observed_data", var_names=target_data, filter_vars=filter_vars, coords=coords
        )

    if independent_data is None:
        independent_data = process_group_variables_coords(
            dt, group="constant_data", var_names="x", filter_vars=filter_vars, coords=coords
        )
    elif isinstance(independent_data, str):
        independent_data = process_group_variables_coords(
            dt,
            group="constant_data",
            var_names=independent_data,
            filter_vars=filter_vars,
            coords=coords,
        )

    (target_var,) = target_data.data_vars
    (independent_var,) = independent_data.data_vars
    # pred_target_var = target_var
    pred_independent_var = independent_var

    if prediction_data is None:
        prediction_data = process_group_variables_coords(
            dt, group=prediction_group, var_names=target_var, filter_vars=filter_vars, coords=coords
        )
    elif isinstance(prediction_data, str):
        # pred_target_var = prediction_data
        prediction_data = process_group_variables_coords(
            dt,
            group=prediction_group,
            var_names=prediction_data,
            filter_vars=filter_vars,
            coords=coords,
        )

    if ci_mean_independent_data is None:
        if prediction_group == "predictions":
            ci_mean_independent_data = process_group_variables_coords(
                dt,
                group="predictions_constant_data",
                var_names=independent_var,
                filter_vars=filter_vars,
                coords=coords,
            )
        else:
            ci_mean_independent_data = independent_data
    elif isinstance(ci_mean_independent_data, str):
        if prediction_group == "predictions":
            pred_independent_var = ci_mean_independent_data
            ci_mean_independent_data = process_group_variables_coords(
                dt,
                group="predictions_constant_data",
                var_names=ci_mean_independent_data,
                filter_vars=filter_vars,
                coords=coords,
            )
        else:
            ci_mean_independent_data = independent_data

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    bg_color = plot_bknd.get_background_color()
    _, contrast_gray_color = get_contrast_colors(bg_color=bg_color, gray_flag=True)
    if plot_collection is None:
        pc_kwargs.setdefault("cols", None)
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, prediction_data)
        plot_collection = PlotCollection.wrap(
            prediction_data,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals.setdefault("line", plot_collection.aes_set)

    # calculations for credible interval
    if ci_kind == "eti":
        ci = prediction_data.azstats.eti(prob=ci_prob, **stats.get("credible_interval", {}))
    elif ci_kind == "hdi":
        ci = prediction_data.azstats.hdi(prob=ci_prob, **stats.get("credible_interval", {}))
    ci_upper = ci.y.sel(ci_bound="upper").values
    ci_lower = ci.y.sel(ci_bound="lower").values

    mean_data = prediction_data.mean(dim=["chain", "draw"])
    pred_x_vals = ci_mean_independent_data[pred_independent_var].values
    colors = plot_bknd.get_default_aes("color", 2, {})

    ci_data = np.stack([ci_lower, ci_upper], axis=1)

    if smooth is True:
        smooth_kwargs = {}
        smooth_kwargs.setdefault("window_length", 55)
        smooth_kwargs.setdefault("polyorder", 2)
        x_data = np.linspace(pred_x_vals.min(), pred_x_vals.max(), 200)
        x_data[0] = (x_data[0] + x_data[1]) / 2
        hdi_interp = griddata(pred_x_vals, ci_data, x_data)
        y_data = savgol_filter(hdi_interp, axis=0, **smooth_kwargs)
    else:
        idx = np.argsort(pred_x_vals)
        x_data = pred_x_vals[idx]
        y_data = ci_data[idx]

    ci_lower = y_data[:, 0]
    ci_upper = y_data[:, 1]

    # upper and lower lines of credible interval
    line_kwargs = copy(visuals.get("line", {}))
    if line_kwargs is not False:
        _, _, line_ignore = filter_aes(plot_collection, aes_by_visuals, "line", sample_dims)
        plot_collection.map(
            line_xy,
            "lower_ci_line",
            x=x_data,
            y=ci_lower,
            ignore_aes=line_ignore,
            **line_kwargs,
        )

        plot_collection.map(
            line_xy,
            "upper_ci_line",
            x=x_data,
            y=ci_upper,
            ignore_aes=line_ignore,
            **line_kwargs,
        )

    # data prepration for `fill` visual
    data = np.stack([x_data, ci_lower, ci_upper], axis=0)
    da = xr.DataArray(
        data,
        dims=["kwarg", "index"],
        coords={
            "kwarg": ["x", "y_bottom", "y_top"],
            "index": np.arange(len(x_data)),
        },
        name="ci_bounds",
    )

    # fill between lines of credible interval
    fill_kwargs = copy(visuals.get("fill", {}))
    if fill_kwargs is not False:
        _, _, fill_ignore = filter_aes(plot_collection, aes_by_visuals, "fill", sample_dims)
        plot_collection.map(
            fill_between_y,
            "fill",
            data=da,
            ignore_aes=fill_ignore,
            color=colors[1],
            alpha=0.4,
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
            x=pred_x_vals,
            y=mean_data.y.values,
            ignore_aes=mean_line_ignore,
            **mean_line_kwargs,
        )

    # scatter plot
    original_scatter_kwargs = copy(visuals.get("scatter", {}))
    if original_scatter_kwargs is not False:
        _, _, scatter_ignore = filter_aes(plot_collection, aes_by_visuals, "scatter", sample_dims)
        plot_collection.map(
            scatter_xy,
            "scatter",
            x=independent_data[independent_var].values,
            y=target_data[target_var].values,
            alpha=0.3,
            color=contrast_gray_color,
            width=0,
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
            text="Predictor",
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
            text="Outcome",
            ignore_aes=ylabel_ignore,
            **ylabel_kwargs,
        )

    return plot_collection
