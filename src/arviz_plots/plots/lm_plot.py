"""lm plot code."""

import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal

import arviz_stats as azs
import numpy as np
import xarray as xr
from arviz_base import extract, rcParams
from arviz_base.labels import BaseLabeller
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    ci_line_y,
    fill_between_y,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
)


def plot_lm(
    dt,
    x=None,
    y=None,
    y_obs=None,
    plot_dim=None,
    smooth=True,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    ci_kind=None,
    ci_prob=None,
    point_estimate=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "pe_line",
            "ci_band",
            "ci_bounds",
            "ci_line_y",
            "observed_scatter",
            "xlabel",
            "ylabel",
        ],
        list[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "pe_line",
            "ci_band",
            "ci_bounds",
            "ci_line_y",
            "observed_scatter",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[
        Literal["credible_interval", "point_estimate", "smooth"],
        Mapping[str, Any] | xr.Dataset,
    ] = None,
    **pc_kwargs,
):
    """Posterior predictive and mean plots for regression-like data.

    Parameters
    ----------
    dt : DataTree
        Input data
    x : str, optional
        Independent variable. If None, use the first variable in group.
        Data will be taken from the constant_data group unless the `group` argument is
        "predictions" in which case it is taken from the predictions_constant_data group.
    y : str optional
        Response variable or linear term. If None, use the first variable in observed_data group.
    y_obs : str or DataArray, optional
        Observed response variable. If None, use `y`.
    plot_dim : str, optional
        Dimension to be represented as the x axis. Defaults to the first dimension
        in the data for `x`. It should be present in the data for `y` too.
    smooth : bool, default True
        If True, apply a Savitzky-Golay filter to smooth the lines.
    filter_vars: {None, “like”, “regex”}, default None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
        It is used for any of y, x, y_pred, and x_pred if they are strings or lists of strings.
    group : str, default "posterior_predictive"
        Group to use for plotting.
    coords : mapping, optional
        Coordinates to use for plotting.
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    ci_kind : {"hdi", "eti"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_prob : float or array-like of floats, optional
        Indicates the probabilities that should be contained within the plotted credible intervals.
        Defaults to ``rcParams["stats.ci_prob"]``
    point_estimate : {"mean", "median","mode"}, optional
        Which point_estimate to use for the line. Defaults to ``rcParams["stats.point_estimate"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.
        By default, there are no aesthetic mappings at all
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * pe_line-> passed to :func:`~.visuals.line_xy`.

          Represents the mean, median, or mode of the predictions, E(y|x), or of the
          linear predictor, E(η|x).

        * ci_band -> passed to :func:`~.visuals.fill_between_y`.

          Represents a credible interval for E(y|x) or E(η|x).

        * ci_bounds -> passed to :func:`~.visuals.line_xy`. Defaults to False

          Represents the upper and lower bounds of a credible interval for E(y|x) or E(η|x).
          This is similar to ci_band, but uses lines for the boundaries instead of a
          filled area.

        * ci_line_y -> passed to :func:`~.visuals.ci_line_y`. Defaults to False

          This is intended for categorical x values or discrete variables with
          few unique values of x for which ci_band or ci_bounds do not work well.

        * observed_scatter -> passed to :func:`~.visuals.scatter_xy`.

          Represents the observed data points.

        * xlabel -> passed to :func:`~.visuals.labelled_x`.
        * ylabel -> passed to :func:`~.visuals.labelled_y`.

    stats : mapping, optional
        Valid keys are:

        * credible_interval -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`


    Returns
    -------
    PlotMatrix

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
        ci_kind = rcParams["stats.ci_kind"]

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
    if point_estimate is None:
        point_estimate = rcParams["stats.point_estimate"]
    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if point_estimate not in ("mean", "median", "mode"):
        raise ValueError("point_estimate must be one of 'mean', 'median', or 'mode'")

    obs_data = get_group(dt, "observed_data")
    if y is None:
        y = list(obs_data.data_vars)[:1]
    elif isinstance(y, str):
        y = [y]

    const_data = get_group(dt, "constant_data")
    if x is None:
        x = list(const_data.data_vars)[:1]
    elif isinstance(x, str):
        x = [x]

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if group in ["posterior", "prior", "posterior_predictive", "prior_predictive"]:
        x_pred = process_group_variables_coords(
            dt,
            group="constant_data",
            var_names=x,
            filter_vars=filter_vars,
            coords=coords,
        )
    elif group == "predictions":
        x_pred = process_group_variables_coords(
            dt,
            group="predictions_constant_data",
            var_names=x,
            filter_vars=filter_vars,
            coords=coords,
        )

    y_pred = process_group_variables_coords(
        dt,
        group=group,
        var_names=y,
        filter_vars=filter_vars,
        coords=coords,
    )

    observed_x = process_group_variables_coords(
        dt,
        group="constant_data",
        var_names=x,
        filter_vars=filter_vars,
        coords=coords,
    )

    if y_obs is None:
        y_obs = y
    observed_y = extract(dt, group="observed_data", var_names=y_obs, combined=False)

    if isinstance(ci_prob, (list | tuple | np.ndarray)):
        x_pred = x_pred.expand_dims(dim={"prob": ci_prob})

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if plot_collection is None:
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if isinstance(ci_prob, (list | tuple | np.ndarray)):
            if "alpha" not in pc_kwargs["aes"]:
                pc_kwargs["aes"].setdefault("alpha", ["prob"])
                len_probs = len(ci_prob)
                pc_kwargs["alpha"] = np.logspace(1, (1 / len_probs), len_probs) / 10
            else:
                warnings.warn(
                    "When multiple credible intervals are plotted, "
                    "it is recommended to map 'alpha' aesthetic to 'prob' "
                    "dimension to differentiate between intervals.",
                )

        pc_kwargs["aes"].setdefault("color", ["__variable__"])
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, x_pred)
        plot_collection = PlotCollection.wrap(
            x_pred,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals.setdefault(
        "central_line", plot_collection.aes_set.difference({"alpha", "color"})
    )
    if isinstance(ci_prob, (list | tuple | np.ndarray)):
        aes_by_visuals.setdefault("ci_line_y", {"alpha"})
        aes_by_visuals.setdefault(
            "ci_band", set(aes_by_visuals.get("ci_band", {})).union({"color", "alpha"})
        )
    else:
        aes_by_visuals.setdefault(
            "ci_band", set(aes_by_visuals.get("ci_band", {})).union({"color"})
        )

    # calculations for credible interval
    ci_fun = azs.hdi if ci_kind == "hdi" else azs.eti
    ci_dims, _, fill_ignore = filter_aes(plot_collection, aes_by_visuals, "ci_band", sample_dims)
    if isinstance(ci_prob, (list | tuple | np.ndarray)):
        ci_data = xr.concat(
            [
                ci_fun(
                    y_pred, dim=ci_dims, prob=p, **stats.get("credible_interval", {})
                ).expand_dims(prob=[p])
                for p in ci_prob
            ],
            dim="prob",
        )
    else:
        ci_data = ci_fun(y_pred, dim=ci_dims, prob=ci_prob, **stats.get("credible_interval", {}))

    central_line_dims, _, _ = filter_aes(
        plot_collection, aes_by_visuals, "central_line", sample_dims
    )
    if point_estimate == "mean":
        pe_value = y_pred.mean(dim=central_line_dims, **stats.get("point_estimate", {}))
    elif point_estimate == "median":
        pe_value = y_pred.median(dim=central_line_dims, **stats.get("point_estimate", {}))
    else:
        pe_value = azs.mode(y_pred, dim=central_line_dims, **stats.get("point_estimate", {}))

    ds_combined = combine(
        x_pred, plot_dim, pe_value, ci_data, x, y, smooth, stats.get("smooth", {})
    )

    lines = plot_bknd.get_default_aes("linestyle", 2, {})

    # Plot credible interval bounds
    ci_bounds_kwargs = get_visual_kwargs(visuals, "ci_bounds", False)
    if ci_bounds_kwargs is not False:
        _, ci_bounds_aes, ci_bounds_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ci_bounds", sample_dims
        )
        if "color" not in ci_bounds_aes:
            ci_bounds_kwargs.setdefault("color", contrast_gray_color)
        if "linestyle" not in ci_bounds_aes:
            ci_bounds_kwargs.setdefault("linestyle", lines[1])

        # Plot upper and lower bounds
        plot_collection.map(
            line_xy,
            "ci_bounds_upper",
            x=ds_combined.sel(plot_axis="x"),
            y=ds_combined.sel(plot_axis="y_top"),
            ignore_aes=ci_bounds_ignore,
            **ci_bounds_kwargs,
        )

        plot_collection.map(
            line_xy,
            "ci_bounds_lower",
            x=ds_combined.sel(plot_axis="x"),
            y=ds_combined.sel(plot_axis="y_bottom"),
            ignore_aes=ci_bounds_ignore,
            **ci_bounds_kwargs,
        )

    # credible band
    fill_kwargs = get_visual_kwargs(visuals, "ci_band")
    if fill_kwargs is not False:
        plot_collection.map(
            fill_between_y,
            "ci_band",
            x=ds_combined.sel(plot_axis="x"),
            y_bottom=ds_combined.sel(plot_axis="y_bottom"),
            y_top=ds_combined.sel(plot_axis="y_top"),
            ignore_aes=fill_ignore,
            **fill_kwargs,
        )

    # credible lines
    ci_line_y_kwargs = get_visual_kwargs(visuals, "ci_line_y", False)
    if ci_line_y_kwargs is not False:
        plot_collection.map(
            ci_line_y,
            "ci_line_y",
            data=ds_combined,
            ignore_aes=fill_ignore,
            **ci_line_y_kwargs,
        )

    # point estimate line
    pe_line_kwargs = get_visual_kwargs(visuals, "pe_line")
    if pe_line_kwargs is not False:
        _, pe_line_aes, pe_line_ignore = filter_aes(
            plot_collection, aes_by_visuals, "pe_line", sample_dims
        )
        if "color" not in pe_line_aes:
            pe_line_kwargs.setdefault("color", contrast_color)
        if "alpha" not in pe_line_aes:
            pe_line_kwargs.setdefault("alpha", 0.6)
        plot_collection.map(
            line_xy,
            "pe_line",
            data=ds_combined,
            ignore_aes=pe_line_ignore,
            **pe_line_kwargs,
        )

    # scatter plot
    observed_scatter_kwargs = get_visual_kwargs(visuals, "observed_scatter")
    if observed_scatter_kwargs is not False:
        _, scatter_aes, scatter_ignore = filter_aes(
            plot_collection, aes_by_visuals, "observed_scatter", sample_dims
        )
        if "alpha" not in scatter_aes:
            observed_scatter_kwargs.setdefault("alpha", 0.3)
        if "color" not in scatter_aes:
            observed_scatter_kwargs.setdefault("color", contrast_gray_color)
        if "width" not in scatter_aes:
            observed_scatter_kwargs.setdefault("width", 0)
        plot_collection.map(
            scatter_xy,
            "observed_scatter",
            x=observed_x,
            y=observed_y,
            ignore_aes=scatter_ignore,
            **observed_scatter_kwargs,
        )

    # x-axis label
    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        _, _, xlabel_ignore = filter_aes(plot_collection, aes_by_visuals, "xlabel", sample_dims)
        plot_collection.map(
            labelled_x,
            "xlabel",
            data=x_pred,
            labeller=labeller,
            subset_info=True,
            ignore_aes=xlabel_ignore,
            **xlabel_kwargs,
        )

    # y-axis label
    ylabel_kwargs = get_visual_kwargs(visuals, "ylabel")
    if ylabel_kwargs is not False:
        _, _, ylabel_ignore = filter_aes(plot_collection, aes_by_visuals, "ylabel", sample_dims)
        plot_collection.map(
            labelled_y,
            "ylabel",
            text=y,
            ignore_aes=ylabel_ignore,
            **ylabel_kwargs,
        )

    return plot_collection


# This ended up being overly complicated, we can write functions
# that work on 2d arrays with shape (obs_id, plot_axis) and use `make_ufunc` in arviz-stats
def sort_values_by_x(values):
    for j in np.ndindex(values.shape[:-2]):
        order = np.argsort(values[j][:, 0], axis=-1)
        values[j] = values[j][order, :]
    return values


def smooth_values(values, n_points=200, **smooth_kwargs):
    x_sorted = values[..., 0]
    x_grid = np.linspace(x_sorted.min(axis=-1), x_sorted.max(axis=-1), n_points)
    x_grid[..., 0] = (x_grid[..., 0] + x_grid[..., 1]) / 2
    out_shape = list(values.shape)
    out_shape[-2] = n_points
    values_smoothed = np.zeros(out_shape, dtype=float)
    values_smoothed[..., 0] = x_grid
    for j in np.ndindex(values_smoothed.shape[:-2]):
        for i in range(1, 4):
            y_interp = griddata(x_sorted[j], values[j][:, i], x_grid[j])
            values_smoothed[j][:, i] = savgol_filter(y_interp, axis=0, **smooth_kwargs)
    return values_smoothed


def combine(x_pred, plot_dim, pe_value, ci_data, x_vars, y_vars, smooth, smooth_kwargs):
    """
    Combine and sort x_pred, pe_value, ci_data into a dataset.

    The resulting dataset will have a dimension plot_axis=['x','y','y_bottom','y_top'],
    and will sort each variable by its x values, and optionally smooth along dim_0.
    """
    combined_data = xr.concat(
        (
            x_pred.expand_dims(plot_axis=["x"]),
            pe_value.expand_dims(plot_axis=["y"]).rename(dict(zip(y_vars, x_vars))),
            ci_data.rename(**dict(zip(y_vars, x_vars)), ci_bound="plot_axis").assign_coords(
                plot_axis=["y_bottom", "y_top"]
            ),
        ),
        dim="plot_axis",
        coords="minimal",
    )

    combined_data = xr.apply_ufunc(
        sort_values_by_x,
        combined_data,
        input_core_dims=[[plot_dim, "plot_axis"]],
        output_core_dims=[[plot_dim, "plot_axis"]],
    )

    if smooth:
        smooth_kwargs.setdefault("window_length", 55)
        smooth_kwargs.setdefault("polyorder", 2)
        smooth_kwargs.setdefault("n_points", 200)

        combined_data = xr.apply_ufunc(
            smooth_values,
            combined_data,
            input_core_dims=[[plot_dim, "plot_axis"]],
            output_core_dims=[[f"smoothed_{plot_dim}", "plot_axis"]],
            kwargs=smooth_kwargs,
        )

    return combined_data
