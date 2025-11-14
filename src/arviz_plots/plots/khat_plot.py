"""Plot Pareto tail indices."""

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    annotate_bin_text,
    calculate_khat_bin_edges,
    enable_hover_labels,
    filter_aes,
    format_coords_as_labels,
    get_visual_kwargs,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    annotate_xy,
    hline,
    labelled_title,
    labelled_x,
    labelled_y,
    scatter_xy,
    set_xlim,
    set_xticks,
)


def plot_khat(
    elpd_data,
    threshold=None,
    hover_format="{index}: {label}",
    legend=None,
    color=None,
    marker=None,
    hline_values=None,
    bin_format="{pct:.1f}%",
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "khat",
            "threshold_text",
            "hover",
            "title",
            "xlabel",
            "ylabel",
            "ticks",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "khat",
            "hlines",
            "bin_text",
            "threshold_text",
            "hover",
            "title",
            "xlabel",
            "ylabel",
            "legend",
            "ticks",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    **pc_kwargs,
):
    r"""Plot Pareto tail indices for diagnosing convergence in PSIS-LOO-CV.

    The Generalized Pareto distribution (GPD) is fitted to the largest importance ratios to
    diagnose convergence rates. The shape parameter :math:`\hat{k}` estimates the pre-asymptotic
    convergence rate based on the fractional number of finite moments. Values :math:`\hat{k} > 0.7`
    indicate impractically low convergence rates and unreliable estimates. Details are presented
    in [1]_ and [2]_.

    Parameters
    ----------
    elpd_data : ELPDData
        ELPD data object returned by :func:`arviz_stats.loo` containing Pareto k diagnostics.
    threshold : float, optional
        Highlight khat values above this threshold with annotations. If None, no points
        are highlighted.
    hover_format : str, default ``"{index}: {label}"``
        Format string for hover annotations. Supports ``{index}``, ``{label}``, and ``{value}``.
    legend : bool, optional
        Whether to display a legend when color aesthetics are active. If None, a legend is shown
        when a color mapping is available.
    color : color spec or str, optional
        Color for scatter points when no aesthetic mapping supplies one. If the value matches a
        dimension name, that dimension is mapped to the color aesthetic.
    marker : marker spec or str, optional
        Marker style for scatter points when no aesthetic mapping supplies one. If the value
        matches a dimension name, that dimension is mapped to the marker aesthetic.
    hline_values : sequence of float, optional
        Custom horizontal line positions. Defaults to [0.0, 0.7, 1.0].
    bin_format : str, default ``"{pct:.1f}%"``
        Format string for bin percentages. Supports ``{count}`` and ``{pct}`` placeholders.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
        Plotting backend to use. Defaults to ``rcParams["plot.backend"]``.
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str or False}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

        By default:

        * khat -> uses all available aesthetic mappings
        * threshold_text -> uses no aesthetic mappings
        * hover -> uses no aesthetic mappings
        * title -> uses no aesthetic mappings
        * xlabel -> uses no aesthetic mappings
        * ylabel -> uses no aesthetic mappings
        * ticks -> uses no aesthetic mappings

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * khat -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * hlines -> passed to :func:`~arviz_plots.visuals.hline`, defaults to False
        * bin_text -> passed to :func:`~arviz_plots.visuals.annotate_xy`, defaults to False
        * threshold_text -> passed to :func:`~arviz_plots.visuals.annotate_xy`
        * hover -> enables interactive hover annotations, defaults to False
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`, defaults to False
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * legend -> passed to :class:`arviz_plots.PlotCollection.add_legend`
        * ticks -> passed to :func:`~arviz_plots.visuals.set_xticks`, defaults to False

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`.

    Returns
    -------
    PlotCollection

    Warnings
    --------
    When using custom markers via the ``visuals`` dict, ensure the marker type is compatible
    with your chosen backend. Not all marker types support separate facecolor and edgecolor
    across different backends.

    Examples
    --------
    The most basic usage plots the Pareto k values from a LOO-CV computation. Each point
    represents one observation, with higher k values indicating less reliable importance
    sampling for that observation.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_khat, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> from arviz_stats import loo
        >>> dt = load_arviz_data("rugby")
        >>> elpd_data = loo(dt, var_name="home_points", pointwise=True)
        >>> plot_khat(elpd_data, figure_kwargs={"figsize": (10, 5)})

    .. minigallery:: plot_khat

    References
    ----------
    .. [1] Vehtari et al. *Practical Bayesian model evaluation using leave-one-out cross-validation
        and WAIC*. Statistics and Computing. 27(5) (2017).
        https://doi.org/10.1007/s11222-016-9696-4. arXiv preprint https://arxiv.org/abs/1507.04544.

    .. [2] Vehtari et al. *Pareto Smoothed Importance Sampling*.
        Journal of Machine Learning Research, 25(72) (2024) https://jmlr.org/papers/v25/19-556.html
        arXiv preprint https://arxiv.org/abs/1507.02646
    """
    if hline_values is None:
        good_k = getattr(elpd_data, "good_k", 0.7)
        hline_values = [0.0, good_k, 1.0]
    else:
        hline_values = list(hline_values)

    visuals = {} if visuals is None else visuals.copy()
    visuals.setdefault("title", False)

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if labeller is None:
        labeller = BaseLabeller()

    if aes_by_visuals is None:
        aes_by_visuals = {}

    if not hasattr(elpd_data, "pareto_k") or elpd_data.pareto_k is None:
        raise ValueError(
            "Could not find 'pareto_k' in the ELPDData object. "
            "Please ensure the LOO computation includes Pareto k diagnostics."
        )

    khat_data = elpd_data.pareto_k
    distribution = khat_data.to_dataset(name="pareto_k")

    n_data_points = khat_data.size
    khat_dims = list(khat_data.dims)
    coord_map = {dim: khat_data.coords[dim] for dim in khat_dims if dim in khat_data.coords}

    khat_flat = np.asarray(khat_data.values).reshape(-1)
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharex", False)
        pc_kwargs["figure_kwargs"].setdefault("sharey", True)

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()

        if isinstance(color, str) and (color in distribution.dims or color in distribution.coords):
            pc_kwargs["aes"]["color"] = [color]
            color = None
        elif color is None and "model" in distribution.dims and "color" not in pc_kwargs["aes"]:
            pc_kwargs["aes"]["color"] = ["model"]

        if isinstance(marker, str) and (
            marker in distribution.dims or marker in distribution.coords
        ):
            pc_kwargs["aes"]["marker"] = [marker]
            marker = None

        use_grid = "rows" in pc_kwargs and pc_kwargs["rows"]

        if use_grid:
            plot_collection = PlotCollection.grid(
                distribution,
                backend=backend,
                **pc_kwargs,
            )
        else:
            pc_kwargs.setdefault("cols", [])
            pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, distribution)

            plot_collection = PlotCollection.wrap(
                distribution,
                backend=backend,
                **pc_kwargs,
            )

    aes_by_visuals.setdefault("khat", plot_collection.aes_set)
    aes_by_visuals.setdefault("threshold_text", [])
    aes_by_visuals.setdefault("hover", [])
    aes_by_visuals.setdefault("title", [])
    aes_by_visuals.setdefault("xlabel", [])
    aes_by_visuals.setdefault("ylabel", [])
    aes_by_visuals.setdefault("ticks", [])

    reduce_dims = [d for d in khat_data.dims if d not in plot_collection.facet_dims]
    has_facets = bool(reduce_dims and plot_collection.facet_dims)

    if has_facets:
        reduce_size = int(np.prod([khat_data.sizes[d] for d in reduce_dims]))
        x_positions_per_facet = np.arange(reduce_size).reshape(
            [khat_data.sizes[d] for d in reduce_dims]
        )
        xdata = xr.DataArray(
            x_positions_per_facet,
            dims=reduce_dims,
            coords={d: khat_data.coords[d] for d in reduce_dims if d in khat_data.coords},
        ).broadcast_like(khat_data)
    else:
        x_positions = (
            np.arange(n_data_points).reshape(khat_data.shape)
            if n_data_points
            else np.zeros(khat_data.shape, dtype=float)
        )
        xdata = xr.DataArray(x_positions, dims=khat_dims, coords=coord_map, name="pareto_k")

    x_flat = np.asarray(xdata.values).reshape(-1)
    x_min = x_flat.min() if x_flat.size else 0.0

    x_dataset = xr.Dataset({"pareto_k": xdata})
    khat_dataset = xr.concat([x_dataset, distribution], dim="plot_axis").assign_coords(
        plot_axis=["x", "y"]
    )

    new_xlim = None
    flat_coord_labels = None
    hover_label_data = None
    if n_data_points and (
        threshold is not None
        or get_visual_kwargs(visuals, "ticks", default=False) is not False
        or get_visual_kwargs(visuals, "hover", default=False) is not False
    ):
        flat_coord_labels = format_coords_as_labels(khat_data, labeller=labeller)
        if flat_coord_labels.size == khat_data.size:
            hover_label_data = xr.DataArray(
                flat_coord_labels.reshape(khat_data.shape),
                dims=khat_dims,
                coords=coord_map,
                name="labels",
            )

    scalar_ds = xr.Dataset({"pareto_k": xr.DataArray(0)})
    khat_kwargs = get_visual_kwargs(visuals, "khat")
    if khat_kwargs is not False:
        _, khat_aes, khat_ignore = filter_aes(plot_collection, aes_by_visuals, "khat", [])

        if "color" not in khat_aes:
            khat_kwargs.setdefault("color", color if color is not None else "C0")

        if "marker" not in khat_aes and marker is not None:
            khat_kwargs.setdefault("marker", marker)

        plot_collection.map(
            scatter_xy,
            "khat",
            data=khat_dataset,
            ignore_aes=khat_ignore,
            **khat_kwargs,
        )

    hlines_kwargs = get_visual_kwargs(visuals, "hlines", default=False)
    if hlines_kwargs is not False and hline_values:

        def _hline_scalar(da, target, **kw):
            scalar_val = da.values.flat[0] if da.size > 0 else 0
            return hline(xr.DataArray(scalar_val), target, **kw)

        for idx, value in enumerate(hline_values):
            hline_kwargs = hlines_kwargs.copy()
            hline_kwargs.setdefault(
                "linestyle", plot_bknd.get_default_aes("linestyle", len(hline_values), {})[idx]
            )
            hline_kwargs.setdefault("color", f"C{idx + 1}")
            hline_kwargs.setdefault("alpha", 0.7)

            hline_data = xr.full_like(khat_data, value)
            hline_dataset = hline_data.to_dataset(name="pareto_k")

            plot_collection.map(
                _hline_scalar,
                f"hline_{idx}",
                data=hline_dataset,
                ignore_aes="all",
                **hline_kwargs,
            )

    bin_text_kwargs = get_visual_kwargs(visuals, "bin_text", default=False)
    if bin_text_kwargs is not False:
        bin_text_kwargs.setdefault("color", "B1")
        bin_text_kwargs.setdefault("horizontal_align", "center")

        bin_edges = calculate_khat_bin_edges(khat_flat, list(hline_values))

        if bin_edges is not None and n_data_points:
            _, _, bin_text_ignore = filter_aes(plot_collection, aes_by_visuals, "bin_text", [])

            if reduce_dims:
                x_max_per_facet = xdata.max(dim=reduce_dims)
            else:
                x_max_per_facet = xdata.max()

            span = x_flat.max() - x_flat.min() if x_flat.size else 1.0
            span = max(span, 1.0)
            x_margin = max(0.05 * span, 0.5)

            if plot_collection.facet_dims:
                x_text_per_facet = x_max_per_facet + x_margin
                # We need to extract the scalar value here for Bokeh compatibility
                new_xlim_max = float(x_max_per_facet.max().item()) + x_margin
            else:
                x_text_per_facet = x_flat.max() + x_margin if x_flat.size else x_margin
                new_xlim_max = x_text_per_facet + x_margin

            new_xlim = (x_min, new_xlim_max)

            num_bins = len(bin_edges) - 1
            bin_edges_arr = np.array(bin_edges)
            y_positions = (bin_edges_arr[:-1] + bin_edges_arr[1:]) / 2

            def compute_bin_counts(data_slice):
                flat_data = np.asarray(data_slice).reshape(-1)
                if flat_data.size == 0:
                    return np.zeros(num_bins, dtype=int)
                counts, _ = np.histogram(flat_data, bins=bin_edges)
                return counts

            if reduce_dims:
                counts_per_facet = xr.apply_ufunc(
                    compute_bin_counts,
                    khat_data,
                    input_core_dims=[reduce_dims],
                    output_core_dims=[["bin"]],
                    vectorize=True,
                )
                n_per_facet = khat_data.count(dim=reduce_dims)
            else:
                counts_per_facet = xr.DataArray(compute_bin_counts(khat_data), dims=["bin"])
                n_per_facet = khat_data.size

            for i in range(num_bins):
                bin_counts = (
                    counts_per_facet.isel(bin=i)
                    if "bin" in counts_per_facet.dims
                    else counts_per_facet[i]
                )

                plot_collection.map(
                    annotate_bin_text,
                    f"bin_{i}",
                    data=distribution,
                    x=x_text_per_facet,
                    y=y_positions[i],
                    count_da=bin_counts,
                    n_da=n_per_facet,
                    bin_format=bin_format,
                    ignore_aes=bin_text_ignore,
                    **bin_text_kwargs,
                )

    threshold_text_kwargs = get_visual_kwargs(visuals, "threshold_text")
    if (
        threshold_text_kwargs is not False
        and threshold is not None
        and flat_coord_labels is not None
    ):
        _, _, threshold_text_ignore = filter_aes(
            plot_collection, aes_by_visuals, "threshold_text", []
        )
        threshold_text_kwargs.setdefault("color", "B1")
        threshold_text_kwargs.setdefault("vertical_align", "bottom")
        threshold_text_kwargs.setdefault("horizontal_align", "center")

        mask = np.asarray(khat_data > threshold).reshape(-1)
        indices = np.flatnonzero(mask)

        for flat_idx in indices:
            label_text = str(flat_coord_labels[flat_idx])
            plot_collection.map(
                annotate_xy,
                f"threshold_{flat_idx}",
                data=scalar_ds,
                x=x_flat[flat_idx],
                y=khat_flat[flat_idx],
                text=label_text,
                ignore_aes=threshold_text_ignore,
                **threshold_text_kwargs,
            )

    ticks_kwargs = get_visual_kwargs(visuals, "ticks", default=False)
    if ticks_kwargs is not False and flat_coord_labels is not None:
        if flat_coord_labels.size:
            ticks_kwargs.setdefault("rotation", 45)

            plot_collection.map(
                set_xticks,
                "ticks",
                data=scalar_ds,
                values=x_flat.tolist(),
                labels=[str(label) for label in flat_coord_labels],
                ignore_aes="all",
                store_artist=False,
                **ticks_kwargs,
            )

    title_kwargs = get_visual_kwargs(visuals, "title")
    if title_kwargs is not False:
        _, title_aes, title_ignore = filter_aes(plot_collection, aes_by_visuals, "title", [])
        if "color" not in title_aes:
            title_kwargs.setdefault("color", "B1")

        def title_coords_only(da, target, sel=None, isel=None, **kw):
            text = labeller.sel_to_str(sel, isel) if (sel or isel) else None
            return labelled_title(da, target, text=text, **kw)

        plot_collection.map(
            title_coords_only,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            **title_kwargs,
        )

    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        _, xlabel_aes, xlabel_ignore = filter_aes(plot_collection, aes_by_visuals, "xlabel", [])

        if "color" not in xlabel_aes:
            xlabel_kwargs.setdefault("color", "B1")
        xlabel_kwargs.setdefault("text", "Data Point")

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=xlabel_ignore,
            subset_info=True,
            **xlabel_kwargs,
        )

    ylabel_kwargs = get_visual_kwargs(visuals, "ylabel")
    if ylabel_kwargs is not False:
        _, ylabel_aes, ylabel_ignore = filter_aes(plot_collection, aes_by_visuals, "ylabel", [])
        if "color" not in ylabel_aes:
            ylabel_kwargs.setdefault("color", "B1")
        ylabel_kwargs.setdefault("text", "Shape parameter k")

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=ylabel_ignore,
            subset_info=True,
            **ylabel_kwargs,
        )

    if legend is not False:
        legend_kwargs = get_visual_kwargs(visuals, "legend")
        if legend_kwargs is not False and "color" in plot_collection.aes.children:
            color_mapping = plot_collection.aes["color"].data_vars.get("mapping")
            if color_mapping is not None:
                legend_kwargs.setdefault("dim", list(color_mapping.dims) or ["color"])
                plot_collection.add_legend(**legend_kwargs)

    if new_xlim is not None:
        plot_collection.map(
            set_xlim,
            "xlim",
            data=distribution,
            ignore_aes="all",
            store_artist=False,
            limits=new_xlim,
        )

    hover_kwargs = get_visual_kwargs(visuals, "hover", default=False)
    if hover_kwargs is not False and flat_coord_labels is not None:
        if hover_label_data is None:
            hover_label_data = xr.DataArray(
                flat_coord_labels.reshape(khat_data.shape),
                dims=khat_dims,
                coords=coord_map,
                name="labels",
            )

        enable_hover_labels(
            backend,
            plot_collection,
            hover_format,
            labels=hover_label_data,
            colors=None,
            values=khat_data,
        )
    return plot_collection
