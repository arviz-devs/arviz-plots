"""Time series plot code."""

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
    vline,
)


def _combine_xy(x_da, y_ds, time_dim):
    """Combine x and y into a Dataset with a `plot_axis` dimension.

    Each variable in `y_ds` becomes a DataArray with `plot_axis=['x','y']`
    where the 'x' slice holds the time values and the 'y' slice holds the
    variable values. This is the format expected by :func:`~.visuals.line_xy`
    and :func:`~.visuals.scatter_xy`.

    Parameters
    ----------
    x_da : DataArray
        1-D time axis values with dimension `time_dim`.
    y_ds : Dataset
        Variable values over the time axis.
    time_dim : str
        Name of the time dimension shared by `x_da` and `y_ds`.

    Returns
    -------
    Dataset
        Dataset where each variable has an extra `plot_axis` dimension.
    """
    result = {}
    for var in y_ds.data_vars:
        # Align x to the same shape as y along time_dim when necessary
        x_broadcast = x_da.broadcast_like(y_ds[var])
        combined = xr.concat(
            [
                x_broadcast.expand_dims(plot_axis=["x"]),
                y_ds[var].expand_dims(plot_axis=["y"]),
            ],
            dim="plot_axis",
        )
        result[var] = combined
    return xr.Dataset(result)


def plot_ts(
    dt,
    *,
    y=None,
    x=None,
    y_hat=None,
    y_holdout=None,
    y_forecasts=None,
    x_holdout=None,
    num_samples=50,
    plot_dim=None,
    filter_vars=None,
    coords=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "observed_line",
            "posterior_predictive",
            "observed_scatter",
            "forecast",
            "vline",
            "xlabel",
            "ylabel",
        ],
        list[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "observed_line",
            "posterior_predictive",
            "observed_scatter",
            "forecast",
            "vline",
            "xlabel",
            "ylabel",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    **pc_kwargs,
):
    """Plot time series data with optional posterior predictive and holdout support.

    Visualizes observed time series data, posterior predictive sample trajectories,
    holdout observations, and forecast trajectories. A vertical line marks the
    train/holdout boundary when holdout data is provided.

    Parameters
    ----------
    dt : DataTree
        Input data. Should contain at minimum an ``observed_data`` group with
        the time series variable specified by `y`.
    y : str or sequence of str, optional
        Variable name(s) from ``observed_data`` to plot on the y-axis (pre-holdout).
        If ``None``, uses the first variable in ``observed_data``.
    x : str, optional
        Name of the coordinate or variable from ``constant_data`` to use as the
        time axis (x-axis) for the pre-holdout segment. If ``None``, uses the
        first dimension coordinate values of the `y` data.
    y_hat : str, optional
        Variable name from ``posterior_predictive`` representing in-sample
        posterior predictive draws. A random subset of ``num_samples`` draws
        are plotted as semi-transparent lines.
    y_holdout : str, optional
        Variable name from ``observed_data`` representing observed data after
        the holdout split point. Plotted as scatter points.
    y_forecasts : str, optional
        Variable name from ``posterior_predictive`` representing out-of-sample
        forecast draws after the holdout split. A random subset of
        ``num_samples`` draws are plotted as semi-transparent lines.
    x_holdout : str, optional
        Name of the coordinate or variable from ``constant_data`` to use as
        the time axis for the holdout segment. If ``None`` and holdout data is
        present, the first dimension coordinate of ``y_holdout`` is used.
    num_samples : int, default 50
        Number of posterior predictive / forecast sample trajectories to draw.
    plot_dim : str, optional
        Name of the dimension representing the time axis in `y` data.
        Defaults to the first non-sample dimension of `y`.
    filter_vars : {None, "like", "regex"}, default None
        If None, interpret `y` as exact variable names.
        If "like", interpret `y` as substrings of variable names.
        If "regex", interpret `y` as regular expressions on variable names.
    coords : dict, optional
        Coordinates to select before plotting.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce when extracting posterior predictive samples.
        Defaults to ``rcParams["data.sample_dims"]``.
    plot_collection : PlotCollection, optional
        Existing PlotCollection to use. If ``None``, a new one is created.
    backend : {"matplotlib", "bokeh", "plotly"}, optional
        Plotting backend. Defaults to ``rcParams["plot.backend"]``.
    labeller : labeller, optional
        Labeller instance for axis labels.
        Defaults to :class:`~arviz_base.labels.BaseLabeller`.
    aes_by_visuals : mapping, optional
        Mapping from visual name to list of aesthetics that should use the
        plot collection's mapping. Valid keys are the same as for `visuals`.
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * ``observed_line`` -> passed to :func:`~.visuals.line_xy`.

          Line showing observed data before the holdout split.

        * ``posterior_predictive`` -> passed to :func:`~.visuals.line_xy`.

          Lines showing ``num_samples`` random posterior predictive trajectories
          from ``y_hat``. Defaults to ``False`` when ``y_hat`` is ``None``.

        * ``observed_scatter`` -> passed to :func:`~.visuals.scatter_xy`.

          Scatter points showing ``y_holdout`` observations after the split.
          Defaults to ``False`` when ``y_holdout`` is ``None``.

        * ``forecast`` -> passed to :func:`~.visuals.line_xy`.

          Lines showing ``num_samples`` random forecast trajectories from
          ``y_forecasts``. Defaults to ``False`` when ``y_forecasts`` is ``None``.

        * ``vline`` -> passed to :func:`~.visuals.vline`.

          Vertical dashed line marking the train/holdout boundary.
          Defaults to ``False`` when no holdout data is provided.

        * ``xlabel`` -> passed to :func:`~.visuals.labelled_x`.

        * ``ylabel`` -> passed to :func:`~.visuals.labelled_y`.

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`.

    Returns
    -------
    PlotCollection

    Examples
    --------
    Basic time series plot showing observed data with posterior predictive samples.

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import xarray as xr
        >>> import arviz_plots as azp
        >>> azp.style.use("arviz-variat")
        >>> rng = np.random.default_rng(42)
        >>> n_time, n_chain, n_draw = 30, 2, 200
        >>> obs = xr.Dataset({"y": (["time"], rng.normal(0, 1, n_time))})
        >>> pp = xr.Dataset(
        ...     {"y": (["chain", "draw", "time"],
        ...            rng.normal(0, 1, (n_chain, n_draw, n_time)))}
        ... )
        >>> dt = xr.DataTree.from_dict(
        ...     {"observed_data": obs, "posterior_predictive": pp}
        ... )
        >>> pc = azp.plot_ts(dt, y="y", y_hat="y")

    .. minigallery:: plot_ts

    """
    # --- defaults ---
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)

    if visuals is None:
        visuals = {}
    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()
    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend
    if labeller is None:
        labeller = BaseLabeller()

    # --- resolve y (observed, pre-holdout) ---
    obs_data = get_group(dt, "observed_data")
    if y is None:
        y = list(obs_data.data_vars)[:1]
    elif isinstance(y, str):
        y = [y]

    observed = process_group_variables_coords(
        dt,
        group="observed_data",
        var_names=y,
        filter_vars=filter_vars,
        coords=coords,
    )

    # Determine the time (plot) dimension
    obs_dims = [d for d in observed.dims if d not in sample_dims]
    if plot_dim is None:
        if obs_dims:
            plot_dim = obs_dims[0]
        else:
            raise ValueError(
                "Could not automatically determine `plot_dim`. "
                "Please pass `plot_dim` explicitly."
            )
    elif plot_dim not in observed.dims:
        raise ValueError(
            f"Dimension '{plot_dim}' given as `plot_dim` is not present in "
            f"the observed data. Available dimensions: {list(observed.dims)}"
        )

    # --- resolve x (time axis values, pre-holdout) ---
    if x is not None:
        const_data = get_group(dt, "constant_data", allow_missing=True)
        if const_data is not None and x in const_data:
            x_values = const_data[x].values
        else:
            x_values = observed[plot_dim].values
    else:
        x_values = observed[plot_dim].values

    x_da = xr.DataArray(x_values, dims=[plot_dim])

    # --- prepare posterior predictive samples (pre-holdout) ---
    pp_subset = None
    if y_hat is not None:
        pp_group = get_group(dt, "posterior_predictive", allow_missing=True)
        if pp_group is not None and y_hat in pp_group:
            pp_data = process_group_variables_coords(
                dt,
                group="posterior_predictive",
                var_names=[y_hat],
                filter_vars=filter_vars,
                coords=coords,
            )
            # Stack sample dims into a single "sample" dimension and select subset
            avail_sample_dims = [d for d in sample_dims if d in pp_data.dims]
            if avail_sample_dims:
                pp_data = pp_data.stack(sample=avail_sample_dims)
            n_total = pp_data.sizes.get("sample", 1)
            rng = np.random.default_rng(0)
            idx = rng.choice(n_total, size=min(num_samples, n_total), replace=False)
            pp_subset = pp_data.isel(sample=idx)
            # Rename from y_hat name to y[0] for alignment with the plot collection
            if y_hat in pp_subset and y_hat != y[0]:
                pp_subset = pp_subset.rename_vars({y_hat: y[0]})

    # --- prepare forecast samples (post-holdout) ---
    fc_subset = None
    if y_forecasts is not None:
        pp_group = get_group(dt, "posterior_predictive", allow_missing=True)
        if pp_group is not None and y_forecasts in pp_group:
            fc_data = process_group_variables_coords(
                dt,
                group="posterior_predictive",
                var_names=[y_forecasts],
                filter_vars=filter_vars,
                coords=coords,
            )
            avail_sample_dims = [d for d in sample_dims if d in fc_data.dims]
            if avail_sample_dims:
                fc_data = fc_data.stack(sample=avail_sample_dims)
            n_total = fc_data.sizes.get("sample", 1)
            rng_fc = np.random.default_rng(1)
            idx_fc = rng_fc.choice(n_total, size=min(num_samples, n_total), replace=False)
            fc_subset = fc_data.isel(sample=idx_fc)
            if y_forecasts in fc_subset and y_forecasts != y[0]:
                fc_subset = fc_subset.rename_vars({y_forecasts: y[0]})

    # --- build PlotCollection ---
    # Use the layout data: if we have PP samples, use them so the 'sample' dim
    # is part of the collection's aes (overlay_ppc); otherwise use observed.
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("color", ["__variable__"])
        if pp_subset is not None:
            pc_kwargs["aes"].setdefault("overlay_ppc", ["sample"])
            layout_data = pp_subset
        else:
            layout_data = observed
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, layout_data)
        plot_collection = PlotCollection.wrap(
            layout_data,
            backend=backend,
            **pc_kwargs,
        )

    # --- aes_by_visuals defaults ---
    aes_by_visuals.setdefault("observed_line", plot_collection.aes_set.difference({"color"}))
    aes_by_visuals.setdefault(
        "posterior_predictive",
        set(plot_collection.aes_set).difference({"color", "overlay_ppc"}),
    )
    aes_by_visuals.setdefault(
        "forecast",
        set(plot_collection.aes_set).difference({"color", "overlay_ppc"}),
    )

    # --- posterior predictive sample lines (y_hat) ---
    pp_kwargs = get_visual_kwargs(visuals, "posterior_predictive", False if pp_subset is None else {})
    if pp_kwargs is not False and pp_subset is not None:
        _, _, pp_ignore = filter_aes(
            plot_collection, aes_by_visuals, "posterior_predictive", sample_dims
        )
        pp_kwargs.setdefault("color", "C0")
        pp_kwargs.setdefault("alpha", 0.15)
        pp_combined = _combine_xy(x_da, pp_subset, plot_dim)
        plot_collection.map(
            line_xy,
            "posterior_predictive",
            data=pp_combined,
            ignore_aes=pp_ignore,
            **pp_kwargs,
        )

    # --- observed line (pre-holdout) ---
    obs_line_kwargs = get_visual_kwargs(visuals, "observed_line")
    if obs_line_kwargs is not False:
        _, _, obs_ignore = filter_aes(
            plot_collection, aes_by_visuals, "observed_line", sample_dims
        )
        obs_line_kwargs.setdefault("color", "B2")
        obs_combined = _combine_xy(x_da, observed, plot_dim)
        plot_collection.map(
            line_xy,
            "observed_line",
            data=obs_combined,
            ignore_aes=obs_ignore,
            **obs_line_kwargs,
        )

    # --- holdout block ---
    has_holdout = (y_holdout is not None) or (y_forecasts is not None)

    # Resolve holdout x-axis values
    x_holdout_da = None
    if has_holdout:
        if x_holdout is not None:
            const_data = get_group(dt, "constant_data", allow_missing=True)
            if const_data is not None and x_holdout in const_data:
                x_holdout_values = const_data[x_holdout].values
            else:
                x_holdout_values = None
        elif y_holdout is not None:
            obs_gp = get_group(dt, "observed_data")
            if y_holdout in obs_gp:
                ho_da = obs_gp[y_holdout]
                ho_dims = [d for d in ho_da.dims if d not in sample_dims]
                if ho_dims:
                    ho_dim = ho_dims[0]
                    x_holdout_values = ho_da[ho_dim].values
                    x_holdout_da = xr.DataArray(x_holdout_values, dims=[ho_dim])
                else:
                    x_holdout_values = None
            else:
                x_holdout_values = None
        else:
            x_holdout_values = None

        if x_holdout_da is None and x_holdout_values is not None:
            # Use same plot_dim name so scatter aligns properly
            x_holdout_da = xr.DataArray(x_holdout_values, dims=[plot_dim])

    # --- observed holdout scatter ---
    obs_scatter_kwargs = get_visual_kwargs(
        visuals, "observed_scatter", False if y_holdout is None else {}
    )
    if obs_scatter_kwargs is not False and y_holdout is not None:
        _, _, scatter_ignore = filter_aes(
            plot_collection, aes_by_visuals, "observed_scatter", sample_dims
        )
        holdout_obs = process_group_variables_coords(
            dt,
            group="observed_data",
            var_names=[y_holdout],
            filter_vars=filter_vars,
            coords=coords,
        )
        # Rename to match y[0] so map() aligns with the PlotCollection layout
        if y_holdout in holdout_obs and y_holdout != y[0]:
            holdout_obs = holdout_obs.rename_vars({y_holdout: y[0]})
        obs_scatter_kwargs.setdefault("color", "B2")
        obs_scatter_kwargs.setdefault("alpha", 0.8)
        obs_scatter_kwargs.setdefault("width", 0)
        # Build the holdout x DataArray aligned to the holdout dataset dims
        ho_dim = list(
            d for d in holdout_obs.dims
            if d not in sample_dims and d != "sample"
        )[0] if holdout_obs.dims else plot_dim
        ho_x_da = (
            x_holdout_da
            if x_holdout_da is not None
            else xr.DataArray(holdout_obs[ho_dim].values, dims=[ho_dim])
        )
        ho_combined = _combine_xy(ho_x_da, holdout_obs, ho_dim)
        plot_collection.map(
            scatter_xy,
            "observed_scatter",
            data=ho_combined,
            ignore_aes=scatter_ignore,
            **obs_scatter_kwargs,
        )

    # --- forecast sample lines (y_forecasts) ---
    fc_kwargs = get_visual_kwargs(visuals, "forecast", False if fc_subset is None else {})
    if fc_kwargs is not False and fc_subset is not None:
        _, _, fc_ignore = filter_aes(
            plot_collection, aes_by_visuals, "forecast", sample_dims
        )
        # Determine forecast time dimension
        fc_time_dims = [
            d for d in fc_subset.dims
            if d not in sample_dims and d != "sample"
        ]
        if fc_time_dims:
            fc_time_dim = fc_time_dims[0]
            fc_x_vals = fc_subset[fc_time_dim].values
            fc_x_da = xr.DataArray(fc_x_vals, dims=[fc_time_dim])
        else:
            fc_time_dim = plot_dim
            fc_x_da = x_holdout_da

        fc_kwargs.setdefault("color", "C1")
        fc_kwargs.setdefault("alpha", 0.15)
        fc_combined = _combine_xy(fc_x_da, fc_subset, fc_time_dim)
        plot_collection.map(
            line_xy,
            "forecast",
            data=fc_combined,
            ignore_aes=fc_ignore,
            **fc_kwargs,
        )

    # --- vertical split line ---
    vline_kwargs = get_visual_kwargs(
        visuals, "vline", False if not has_holdout else {}
    )
    if vline_kwargs is not False and has_holdout and len(x_values) > 0:
        split_x = float(x_values[-1])
        split_da = xr.Dataset(
            {var: xr.DataArray(split_x) for var in observed.data_vars}
        )
        vline_kwargs.setdefault("color", "B3")
        vline_kwargs.setdefault("linestyle", "C1")
        _, _, vline_ignore = filter_aes(
            plot_collection, aes_by_visuals, "vline", sample_dims
        )
        plot_collection.map(
            vline,
            "vline",
            data=split_da,
            ignore_aes=vline_ignore,
            **vline_kwargs,
        )

    # --- x-axis label ---
    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        _, _, xlabel_ignore = filter_aes(
            plot_collection, aes_by_visuals, "xlabel", sample_dims
        )
        plot_collection.map(
            labelled_x,
            "xlabel",
            data=plot_collection.viz["plot"].dataset,
            labeller=labeller,
            subset_info=True,
            ignore_aes=xlabel_ignore,
            **xlabel_kwargs,
        )

    # --- y-axis label ---
    ylabel_kwargs = get_visual_kwargs(visuals, "ylabel")
    if ylabel_kwargs is not False:
        _, _, ylabel_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ylabel", sample_dims
        )
        plot_collection.map(
            labelled_y,
            "ylabel",
            data=plot_collection.viz["plot"].dataset,
            labeller=labeller,
            subset_info=True,
            ignore_aes=ylabel_ignore,
            **ylabel_kwargs,
        )

    return plot_collection
