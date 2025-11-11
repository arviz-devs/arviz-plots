"""lm plot code."""

import warnings
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal

import arviz_stats as azs
import numpy as np
import xarray as xr
from arviz_base import extract, rcParams
from arviz_base.labels import BaseLabeller, MapLabeller
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
    xlabeller=None,
    ylabeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "pe_line",
            "ci_band",
            "ci_bounds",
            "ci_vlines",
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
            "ci_vlines",
            "observed_scatter",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[
        Literal["credible_interval", "pe_line", "smooth"],
        Mapping[str, Any] | xr.Dataset,
    ] = None,
    **pc_kwargs,
):
    """Posterior predictive and mean plots for regression-like data.

    Parameters
    ----------
    dt : DataTree
        Input data
    x : str or sequence of str, optional
        Independent variable. If None, use the first variable in group.
        Data will be taken from the constant_data group unless the `group` argument is
        "predictions" in which case it is taken from the predictions_constant_data group.

        The plots and visuals in the generated ``PlotCollection`` object will use `x` for naming.
    y : str or sequence of str, optional
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
    ci_prob : float or array-like of float, optional
        Indicates the probabilities that should be contained within the plotted credible intervals.
        Defaults to ``rcParams["stats.ci_prob"]``
    point_estimate : {"mean", "median", "mode"}, optional
        Which point_estimate to use for the line. Defaults to ``rcParams["stats.point_estimate"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    xlabeller, ylabeller : labeller, optional
        Labeller for the x and y axes. Will use the `make_label_vert` method of the labeller.
        By default, `xlabeller` is a :class:`~arviz_base.labels.BaseLabeller` and
        `ylabeller` is a :class:`~arviz_base.labels.MapLabeller` that maps values of
        `x` to their respective `y` value given the first ones are used to name things
        in the ``PlotCollection``.
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

        By default, the color is mapped to the variable which is active for the "ci_band" visual.
        If `ci_prob` is not a scalar a mapping from prob->alpha is also added which is
        active for "ci_band" and "ci_vlines" visuals.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * pe_line-> passed to :func:`~.visuals.line_xy`.

          Line that represent the mean, median, or mode of the predictions, E(y|x), or of the
          linear predictor, E(η|x).

        * ci_band -> passed to :func:`~.visuals.fill_between_y`.

          Filled area that represents a credible interval for E(y|x) or E(η|x).

        * ci_bounds -> passed to :func:`~.visuals.line_xy`. Defaults to False

          Lines that represent the upper and lower bounds of a credible interval
          for E(y|x) or E(η|x). This is similar to "ci_band", but uses lines
          for the boundaries instead of a filled area.

        * ci_vlines -> passed to :func:`~.visuals.ci_line_y`. Defaults to False

          This is intended for categorical x values or discrete variables with
          few unique values of x for which ci_band or ci_bounds do not work well.
          Represents the same information as these two visuals but as multiple vertical lines,
          similar to :func:`~arviz_plots.plot_ppc_interval`

        * observed_scatter -> passed to :func:`~.visuals.scatter_xy`.

          Represents the observed data points.

        * xlabel -> passed to :func:`~.visuals.labelled_x`.
        * ylabel -> passed to :func:`~.visuals.labelled_y`.

    stats : mapping, optional
        Valid keys are:

        * credible_interval -> passed to eti or hdi. Affects all 3 visual elements
          related to the credible intervals
        * pe_line -> passed to mean, median or mode
        * smooth -> passed to :func:`scipy.signal.savgol_filter`.
          It also takes an extra ``n_points`` key to control the number of points
          in the interpolation grid that is passed to the smoothing filter.
          Affects the 4 visual elements related to credible intervals or point estimates.

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

    y_to_x_map = dict(zip(y, x))

    if xlabeller is None:
        xlabeller = BaseLabeller()
    if ylabeller is None:
        ylabeller = MapLabeller(
            var_name_map={x_name: y_name for y_name, x_name in y_to_x_map.items()}
        )

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
    if plot_dim is None:
        plot_dim = list(x_pred.dims)[0]
    elif plot_dim not in x_pred.dims:
        raise ValueError(
            f"Dimension '{plot_dim}' given as `plot_dim` argument is not present in x data. "
            f"Present dimensions are {tuple(x_pred.dims)}."
        )

    y_pred = process_group_variables_coords(
        dt,
        group=group,
        var_names=y,
        filter_vars=filter_vars,
        coords=coords,
    ).rename_vars(y_to_x_map)
    if plot_dim not in y_pred.dims:
        error_msg = (
            f"Dimension '{plot_dim}' set as `plot_dim` argument is not present in y data. "
            f"Present dimensions are {tuple(y_pred.dims)}."
        )
        possible_matches = {}
        for xdim, xsize in x_pred.sizes.items():
            matches_i = [ydim for ydim, ysize in y_pred.sizes.items() if ysize == xsize]
            if matches_i:
                possible_matches[xdim] = matches_i
        if possible_matches:
            error_msg += (
                f"\nPossible name mismatches between dimensions in x and y data: {possible_matches}"
            )
        raise ValueError(error_msg)

    observed_x = process_group_variables_coords(
        dt,
        group="constant_data",
        var_names=x,
        filter_vars=filter_vars,
        coords=coords,
    )

    if y_obs is None:
        y_obs = y
    observed_y = extract(
        dt, group="observed_data", var_names=y_obs, combined=False, keep_dataset=True
    )
    if all(var_name in observed_y.data_vars for var_name in y_to_x_map):
        observed_y = observed_y.rename_vars(y_to_x_map)

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
        if isinstance(ci_prob, (list | tuple | np.ndarray)):
            pc_data = x_pred.expand_dims(dim={"prob": ci_prob})
        else:
            pc_data = x_pred

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, pc_data)
        plot_collection = PlotCollection.wrap(
            pc_data,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals.setdefault("pe_line", plot_collection.aes_set.difference({"alpha", "color"}))
    if isinstance(ci_prob, (list | tuple | np.ndarray)):
        aes_by_visuals.setdefault("ci_vlines", {"alpha"})
        aes_by_visuals.setdefault(
            "ci_band", set(aes_by_visuals.get("ci_band", {})).union({"color", "alpha"})
        )
    else:
        aes_by_visuals.setdefault(
            "ci_band", set(aes_by_visuals.get("ci_band", {})).union({"color"})
        )

    # calculations for credible interval
    ci_fun = azs.hdi if ci_kind == "hdi" else azs.eti
    ci_dims, ci_band_aes, ci_band_ignore = filter_aes(
        plot_collection, aes_by_visuals, "ci_band", sample_dims
    )
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

    pe_line_dims, pe_line_aes, pe_line_ignore = filter_aes(
        plot_collection, aes_by_visuals, "pe_line", sample_dims
    )
    if point_estimate == "mean":
        pe_value = y_pred.mean(dim=pe_line_dims, **stats.get("point_estimate", {}))
    elif point_estimate == "median":
        pe_value = y_pred.median(dim=pe_line_dims, **stats.get("point_estimate", {}))
    elif point_estimate == "mode":
        pe_value = azs.mode(y_pred, dim=pe_line_dims, **stats.get("point_estimate", {}))
    else:
        raise ValueError(
            f"'{point_estimate}' is not a valid value for `point_estimate`. "
            "Valid options are mean, median and mode"
        )

    combined_pe, combined_ci = combine_sort_smooth(
        x_pred, plot_dim, pe_value, ci_data, smooth, stats.get("smooth", {})
    )

    # Plot credible interval bounds
    ci_bounds_kwargs = get_visual_kwargs(visuals, "ci_bounds", False)
    if ci_bounds_kwargs is not False:
        _, ci_bounds_aes, ci_bounds_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ci_bounds", sample_dims
        )
        if "color" not in ci_bounds_aes:
            ci_bounds_kwargs.setdefault("color", "B2")
        if "linestyle" not in ci_bounds_aes:
            ci_bounds_kwargs.setdefault("linestyle", "C1")

        # Plot upper and lower bounds
        plot_collection.map(
            line_xy,
            "ci_bounds_upper",
            x=combined_ci.sel(plot_axis="x"),
            y=combined_ci.sel(plot_axis="y_top"),
            ignore_aes=ci_bounds_ignore,
            **ci_bounds_kwargs,
        )

        plot_collection.map(
            line_xy,
            "ci_bounds_lower",
            x=combined_ci.sel(plot_axis="x"),
            y=combined_ci.sel(plot_axis="y_bottom"),
            ignore_aes=ci_bounds_ignore,
            **ci_bounds_kwargs,
        )

    # credible band
    ci_band_kwargs = get_visual_kwargs(visuals, "ci_band")
    if ci_band_kwargs is not False:
        if "color" not in ci_band_aes:
            ci_band_kwargs.setdefault("color", "C0")
        plot_collection.map(
            fill_between_y,
            "ci_band",
            x=combined_ci.sel(plot_axis="x"),
            y_bottom=combined_ci.sel(plot_axis="y_bottom"),
            y_top=combined_ci.sel(plot_axis="y_top"),
            ignore_aes=ci_band_ignore,
            **ci_band_kwargs,
        )

    # credible intervals as multiple vertical lines
    ci_vlines_kwargs = get_visual_kwargs(visuals, "ci_vlines", False)
    if ci_vlines_kwargs is not False:
        _, ci_vlines_aes, ci_vlines_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ci_vlines", sample_dims
        )
        if "color" not in ci_vlines_aes:
            ci_vlines_kwargs.setdefault("color", "C0")
        plot_collection.map(
            ci_line_y,
            "ci_vlines",
            data=combined_ci,
            ignore_aes=ci_vlines_ignore,
            **ci_vlines_kwargs,
        )

    # point estimate line
    pe_line_kwargs = get_visual_kwargs(visuals, "pe_line")
    if pe_line_kwargs is not False:
        if "color" not in pe_line_aes:
            pe_line_kwargs.setdefault("color", "B1")
        if "alpha" not in pe_line_aes:
            pe_line_kwargs.setdefault("alpha", 0.6)
        plot_collection.map(
            line_xy,
            "pe_line",
            data=combined_pe,
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
            observed_scatter_kwargs.setdefault("color", "B2")
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
            data=plot_collection.viz["plot"].dataset,
            labeller=xlabeller,
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
            data=plot_collection.viz["plot"].dataset,
            labeller=ylabeller,
            subset_info=True,
            ignore_aes=ylabel_ignore,
            **ylabel_kwargs,
        )

    return plot_collection


# This ended up being overly complicated, we can write functions
# that work on 2d arrays with shape (obs_id, plot_axis) and use `make_ufunc` in arviz-stats
def _sort_values_by_x(values):
    """Sort values by x along requested dimension for plot_lm purposes."""
    for j in np.ndindex(values.shape[:-2]):
        order = np.argsort(values[j][:, 0], axis=-1)
        values[j] = values[j][order, :]
    return values


def _smooth_values(values, n_points=200, **smooth_kwargs):
    """Smooth values in 1d slices for plot_lm purposes."""
    out_shape = list(values.shape)
    out_shape[-2] = n_points
    values_smoothed = np.empty(out_shape, dtype=float)
    for j in np.ndindex(values_smoothed.shape[:-2]):
        x_sorted_j = values[j][:, 0]
        x_grid = np.linspace(x_sorted_j.min(), x_sorted_j.max(), n_points)
        x_grid[0] = (x_grid[0] + x_grid[1]) / 2
        values_smoothed[j][:, 0] = x_grid
        for i in range(1, values.shape[-1]):
            y_interp = griddata(x_sorted_j, values[j][:, i], x_grid)
            values_smoothed[j][:, i] = savgol_filter(y_interp, axis=0, **smooth_kwargs)
    return values_smoothed


def combine_sort_smooth(x_pred, plot_dim, pe_value, ci_data, smooth, smooth_kwargs):
    """
    Combine and sort x_pred, pe_value, ci_data into two datasets.

    The resulting datasets will have a dimension plot_axis=['x','y'] for the pe related data
    and plot_axis=['x','y_bottom','y_top'] for the ci related data.

    Each variable is sorted by its x values along `plot_dim`, and optionally smoothed along
    this same dimension.

    Separating pe and ci related data ensures pe_data doesn't end up with the `prob` dimension.
    If it did, in the best case scenario we'd en up with multiple perfectly overlapping lines
    in the same plot, or an avoidable and non-sensical error in the worst case scenario.
    """
    combined_pe = xr.concat(
        (
            x_pred.expand_dims(plot_axis=["x"]),
            pe_value.expand_dims(plot_axis=["y"]),
        ),
        dim="plot_axis",
    )

    combined_ci = xr.concat(
        (
            x_pred.expand_dims(plot_axis=["x"]),
            ci_data.rename(ci_bound="plot_axis").assign_coords(plot_axis=["y_bottom", "y_top"]),
        ),
        dim="plot_axis",
        coords="minimal",
    )

    combined_pe = xr.apply_ufunc(
        _sort_values_by_x,
        combined_pe,
        input_core_dims=[[plot_dim, "plot_axis"]],
        output_core_dims=[[plot_dim, "plot_axis"]],
    )

    combined_ci = xr.apply_ufunc(
        _sort_values_by_x,
        combined_ci,
        input_core_dims=[[plot_dim, "plot_axis"]],
        output_core_dims=[[plot_dim, "plot_axis"]],
    )

    if smooth:
        smooth_kwargs.setdefault("window_length", 55)
        smooth_kwargs.setdefault("polyorder", 2)
        smooth_kwargs.setdefault("n_points", 200)

        combined_pe = xr.apply_ufunc(
            _smooth_values,
            combined_pe,
            input_core_dims=[[plot_dim, "plot_axis"]],
            output_core_dims=[[f"smoothed_{plot_dim}", "plot_axis"]],
            kwargs=smooth_kwargs,
        )

        combined_ci = xr.apply_ufunc(
            _smooth_values,
            combined_ci,
            input_core_dims=[[plot_dim, "plot_axis"]],
            output_core_dims=[[f"smoothed_{plot_dim}", "plot_axis"]],
            kwargs=smooth_kwargs,
        )

    return combined_pe, combined_ci
