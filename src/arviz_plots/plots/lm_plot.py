"""lm plot code."""

import warnings
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import arviz_stats as azs
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
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
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    ci_kind=None,
    ci_prob=None,
    line_kind=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "ci_line",
            "central_line",
            "ci_fill",
            "scatter",
            "xlabel",
            "ylabel",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "ci_line",
            "central_line",
            "ci_fill",
            "scatter",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[
        Literal["credible_interval", "point_estimate"],
        Mapping[str, Any] | xr.Dataset,
    ] = None,
    **pc_kwargs,
):
    """Posterior predictive and mean plots for regression-like data..

    Parameters
    ----------
    dt : DataTree
        Input data
    y : str or DataArray, optional
        Target variable. If None (default), the first variable in "observed_data" is used.
    x : str or list of str or DataArray or Dataset, optional
        Independent variable(s). If None (default), all variables in "constant_data" are used.
    y_pred : str or DataArray, optional
        Predicted values.
        If None (default), the variable in the specified group with the same name as y is used.
    x_pred : str or list of str or DataArray or Dataset, optional
        Independent variable(s) for predictions.

        If None (default), and if group is "predictions", all variables corresponding to x data
        in "predictions_constant_data" group are used. If group is "posterior_predictive",
        x is used.
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
    ci_prob : float or list of float, optional
        Indicates the probabilities that should be contained within the plotted credible intervals.
        Defaults to ``rcParams["stats.ci_prob"]``
    line_kind : {"mean", "median","mode"}, optional
        Which point estimate to use for the line. Defaults to ``rcParams["stats.point_estimate"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.
        By default, there are no aesthetic mappings at all
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * ci_line -> passed to :func:`~.visuals.line_xy`. Defaults to False
        * central_line -> passed to :func:`~.visuals.line_xy`.
        * ci_fill -> passed to :func:`~.visuals.fill_between_y`.
        * scatter -> passed to :func:`~.visuals.scatter_xy`.
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
    if line_kind is None:
        line_kind = rcParams["stats.point_estimate"]
    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    obs_data = get_group(dt, "observed_data")
    if y is None:
        y = list(obs_data.data_vars)[0]

    if isinstance(y, xr.Dataset):
        raise TypeError(
            "y can't be a dataset because multiple target variables are not supported yet."
        )

    if not isinstance(y, xr.DataArray):
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

    if isinstance(y_pred, xr.Dataset):
        raise TypeError(
            "y_pred can't be a dataset because multiple target variables are not supported yet."
        )

    if not isinstance(y_pred, xr.DataArray):
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

    if isinstance(ci_prob, Sequence):
        x_with_prob = x.expand_dims(dim={"prob": ci_prob})
    else:
        x_with_prob = x

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if plot_collection is None:
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if isinstance(ci_prob, Sequence):
            if "alpha" not in pc_kwargs["aes"]:
                pc_kwargs["aes"].setdefault("alpha", ["prob"])
                pc_kwargs["alpha"] = np.linspace(0.1, 0.5, len(ci_prob))
            else:
                warnings.warn(
                    "When multiple credible intervals are plotted, "
                    "it is recommended to map 'alpha' aesthetic to 'prob' "
                    "dimension to differentiate between intervals.",
                )

        pc_kwargs["aes"].setdefault("color", ["__variable__"])
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
    aes_by_visuals.setdefault(
        "central_line", plot_collection.aes_set.difference({"alpha", "color"})
    )
    if isinstance(ci_prob, Sequence):
        aes_by_visuals.setdefault("ci_line", {"alpha"})
        aes_by_visuals.setdefault(
            "ci_fill", set(aes_by_visuals.get("ci_fill", {})).union({"color", "alpha"})
        )
    else:
        aes_by_visuals.setdefault(
            "ci_fill", set(aes_by_visuals.get("ci_fill", {})).union({"color"})
        )

    # calculations for credible interval
    ci_fun = azs.hdi if ci_kind == "hdi" else azs.eti
    ci_dims, _, fill_ignore = filter_aes(plot_collection, aes_by_visuals, "ci_fill", sample_dims)
    if isinstance(ci_prob, Sequence):
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
    if line_kind == "mean":
        line_data = y_pred.mean(dim=central_line_dims, **stats.get("point_estimate", {}))
    elif line_kind == "median":
        line_data = y_pred.median(dim=central_line_dims, **stats.get("point_estimate", {}))
    elif line_kind == "mode":
        line_data = azs.mode(y_pred, dim=central_line_dims, **stats.get("point_estimate", {}))

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
            ci_line_kwargs.setdefault("color", "B2")

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
    fill_kwargs = copy(visuals.get("ci_fill", {}))
    if fill_kwargs is not False:
        plot_collection.map(
            fill_between_y,
            "ci_fill",
            x=x_pred,
            y_bottom=ci_lower[target_var],
            y_top=ci_upper[target_var],
            ignore_aes=fill_ignore,
            **fill_kwargs,
        )

    # mean/median/mode line
    central_line_kwargs = copy(visuals.get("central_line", {}))
    if central_line_kwargs is not False:
        _, central_line_aes, central_line_ignore = filter_aes(
            plot_collection, aes_by_visuals, "central_line", sample_dims
        )
        if "color" not in central_line_aes:
            central_line_kwargs.setdefault("color", "B1")
        if "alpha" not in central_line_aes:
            central_line_kwargs.setdefault("alpha", 0.6)
        plot_collection.map(
            line_xy,
            "central_line",
            x=x_pred,
            y=line_data[target_var],
            ignore_aes=central_line_ignore,
            **central_line_kwargs,
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
            original_scatter_kwargs.setdefault("color", "B2")
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
