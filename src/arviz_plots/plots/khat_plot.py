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
    alpha_scaled_colors,
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
    show_hlines=False,
    show_bins=False,
    hover_label=False,
    hover_format="{index}: {label}",
    xlabels=False,
    legend=None,
    color=None,
    hline_values=None,
    bin_format="{pct:.1f}%",
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "khat",
            "hlines",
            "bin_text",
            "threshold_text",
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
    show_hlines : bool, default False
        Show horizontal reference lines at diagnostic thresholds.
    show_bins : bool, default False
        Show the percentage of khat values falling in each bin delimited by reference lines.
    hover_label : bool, default False
        Enable interactive hover annotations when using an interactive backend.
    hover_format : str, default ``"{index}: {label}"``
        Format string for hover annotations. Supports ``{index}``, ``{label}``, and ``{value}``.
    xlabels : bool, default False
        Show coordinate labels as x tick labels.
    legend : bool, optional
        Whether to display a legend when color aesthetics are active. If None, a legend is shown
        when a color mapping is available.
    color : color spec or str, optional
        Color for scatter points when no aesthetic mapping supplies one. If the value matches a
        dimension name, that dimension is mapped to the color aesthetic.
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
        * hlines -> uses no aesthetic mappings
        * bin_text -> uses no aesthetic mappings
        * threshold_text -> uses no aesthetic mappings
        * title -> uses no aesthetic mappings
        * xlabel -> uses no aesthetic mappings
        * ylabel -> uses no aesthetic mappings
        * ticks -> uses no aesthetic mappings

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * khat -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * hlines -> passed to :func:`~arviz_plots.visuals.hline`
        * bin_text -> passed to :func:`~arviz_plots.visuals.annotate_xy`
        * threshold_text -> passed to :func:`~arviz_plots.visuals.annotate_xy`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title` defaults to False
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * legend -> passed to :class:`arviz_plots.PlotCollection.add_legend`
        * ticks -> passed to :func:`~arviz_plots.visuals.set_xticks`

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
        >>> dt = load_arviz_data("radon")
        >>> elpd_data = loo(dt, pointwise=True)
        >>> plot_khat(elpd_data, figure_kwargs={"figsize": (10, 5)})

    We can highlight problematic observations by setting a ``threshold`` and add reference
    lines with ``show_hlines=True`` to visualize the diagnostic boundaries. Using
    ``show_bins=True`` displays the percentage of observations falling into each diagnostic
    category. Note that the ``hline_values`` parameter is independent of the ``threshold``
    parameter. To draw a horizontal line at your custom threshold, you must set both parameters
    explicitly.

    .. plot::
        :context: close-figs

        >>> plot_khat(elpd_data,
        >>>     threshold=0.4,
        >>>     show_hlines=True,
        >>>     show_bins=True,
        >>>     hline_values=[0.0, 0.4, 1.0],
        >>>     visuals={"hlines": {"color":"B1"}},
        >>>     figure_kwargs={"figsize": (10, 5)}
        >>> )

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

    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()

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
    else:
        aes_by_visuals = aes_by_visuals.copy()

    pc_kwargs = dict(pc_kwargs)

    if not hasattr(elpd_data, "pareto_k") or elpd_data.pareto_k is None:
        raise ValueError(
            "Could not find 'pareto_k' in the ELPDData object. "
            "Please ensure the LOO computation includes Pareto k diagnostics."
        )

    khat_data = elpd_data.pareto_k
    distribution = khat_data.to_dataset(name="pareto_k")

    n_data_points = khat_data.size
    khat_dims = list(khat_data.dims)

    flat_coord_labels = format_coords_as_labels(khat_data)
    coord_map = {dim: khat_data.coords[dim] for dim in khat_dims if dim in khat_data.coords}

    if n_data_points:
        x_positions = np.arange(n_data_points).reshape(khat_data.shape)
    else:
        x_positions = np.zeros(khat_data.shape, dtype=float)

    xdata = xr.DataArray(x_positions, dims=khat_dims, coords=coord_map, name="pareto_k")
    x_dataset = xr.Dataset({"pareto_k": xdata})
    khat_dataset = xr.concat([x_dataset, distribution], dim="plot_axis").assign_coords(
        plot_axis=["x", "y"]
    )

    khat_values = np.asarray(khat_data.values).reshape(-1)
    x_flat = np.asarray(x_positions).reshape(-1)
    y_flat = np.asarray(khat_data.values).reshape(-1)
    x_min = x_flat.min() if x_flat.size else 0.0
    x_max = x_flat.max() if x_flat.size else 0.0

    good_k_threshold = hline_values[1] if len(hline_values) > 1 else 0.7
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    scalar_ds = xr.Dataset({"pareto_k": xr.DataArray(0)})

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)
        pc_kwargs["figure_kwargs"].setdefault("sharey", True)

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()

        if isinstance(color, str) and color in distribution.dims:
            pc_kwargs["aes"]["color"] = [color]
            color = None
        elif "model" in distribution.dims and "color" not in pc_kwargs["aes"]:
            pc_kwargs["aes"]["color"] = ["model"]

        pc_kwargs.setdefault("cols", ["__variable__"])
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, distribution)

        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    aes_by_visuals.setdefault("khat", plot_collection.aes_set)
    aes_by_visuals.setdefault("hlines", [])
    aes_by_visuals.setdefault("bin_text", [])
    aes_by_visuals.setdefault("threshold_text", [])
    aes_by_visuals.setdefault("title", [])
    aes_by_visuals.setdefault("xlabel", [])
    aes_by_visuals.setdefault("ylabel", [])
    aes_by_visuals.setdefault("ticks", [])

    point_rgba = None
    new_xlim = None

    khat_kwargs = get_visual_kwargs(visuals, "khat")
    if khat_kwargs is not False:
        _, khat_aes, khat_ignore = filter_aes(plot_collection, aes_by_visuals, "khat", [])

        default_color = khat_kwargs.get("color", color)

        if default_color is None and "color" not in khat_aes:
            default_color = "C0"

        if backend == "matplotlib" and "color" not in khat_aes:
            base_color = khat_kwargs.pop("color", default_color)
            point_rgba = alpha_scaled_colors(base_color, khat_data.values, good_k_threshold)
            khat_kwargs.setdefault("color", point_rgba)
            khat_kwargs.setdefault("zorder", 2)
        elif "color" not in khat_aes and default_color is not None:
            khat_kwargs.setdefault("color", default_color)
            if backend == "matplotlib":
                khat_kwargs.setdefault("zorder", 2)

        plot_collection.map(
            scatter_xy,
            "khat",
            data=khat_dataset,
            ignore_aes=khat_ignore,
            **khat_kwargs,
        )

    if show_hlines and hline_values:
        hlines_kwargs = get_visual_kwargs(visuals, "hlines")

        if hlines_kwargs is not False:
            _, hlines_aes, _ = filter_aes(plot_collection, aes_by_visuals, "hlines", [])
            linestyle_cycle = [":", "-.", "--", "-"]

            for idx, value in enumerate(hline_values):
                h_kwargs = hlines_kwargs.copy()
                if backend == "matplotlib" and "linestyle" not in hlines_aes:
                    h_kwargs.setdefault("linestyle", linestyle_cycle[idx % len(linestyle_cycle)])
                if "color" not in hlines_aes:
                    h_kwargs.setdefault("color", f"C{idx + 1}")
                if "alpha" not in hlines_aes:
                    h_kwargs.setdefault("alpha", 0.7)
                if backend == "matplotlib":
                    h_kwargs.setdefault("zorder", 3)

                h_ds = xr.Dataset({"pareto_k": xr.DataArray(value)})
                plot_collection.map(
                    hline,
                    f"hline_{idx}",
                    data=h_ds,
                    ignore_aes="all",
                    **h_kwargs,
                )

    if show_bins:
        bin_text_kwargs = get_visual_kwargs(visuals, "bin_text")
        if bin_text_kwargs is not False:
            _, bin_text_aes, _ = filter_aes(plot_collection, aes_by_visuals, "bin_text", [])
            if "color" not in bin_text_aes:
                bin_text_kwargs.setdefault("color", "B1")
            bin_text_kwargs.setdefault("horizontal_align", "center")

            bin_edges = calculate_khat_bin_edges(khat_values, [good_k_threshold, 1.0])

            if bin_edges is not None and n_data_points:
                counts, edges = np.histogram(khat_values, bins=bin_edges)
                span = max(1.0, x_max - x_min)
                x_margin = max(0.5, 0.05 * span)
                x_text = x_max + x_margin
                new_xlim = (x_min, x_text + x_margin)

                for bin_idx, count in enumerate(counts):
                    if count == 0:
                        continue
                    lower = edges[bin_idx]
                    upper = edges[bin_idx + 1]

                    if np.isnan(lower) or np.isnan(upper):
                        continue
                    pct = (count / n_data_points * 100) if n_data_points else 0.0
                    label = bin_format.format(count=count, pct=pct)
                    y_pos = 0.5 * (lower + upper)

                    plot_collection.map(
                        annotate_xy,
                        f"bin_{bin_idx}",
                        data=scalar_ds,
                        x=x_text,
                        y=y_pos,
                        text=label,
                        ignore_aes="all",
                        **bin_text_kwargs,
                    )

    if threshold is not None and n_data_points:
        threshold_text_kwargs = get_visual_kwargs(visuals, "threshold_text")

        if threshold_text_kwargs is not False:
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
                    y=y_flat[flat_idx],
                    text=label_text,
                    ignore_aes=threshold_text_ignore,
                    **threshold_text_kwargs,
                )

    if xlabels and x_flat.size and flat_coord_labels.size:
        ticks_kwargs = get_visual_kwargs(visuals, "ticks")

        if ticks_kwargs is not False:
            if backend == "matplotlib" and "rotation" not in ticks_kwargs:
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

        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            labeller=labeller,
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

    legend_kwargs = get_visual_kwargs(visuals, "legend", default=None)
    if legend is False:
        legend_kwargs = False
    elif legend_kwargs is None:
        legend_kwargs = {}
    if legend_kwargs is not False and "color" in plot_collection.aes.children:
        color_mapping = plot_collection.aes["color"].data_vars.get("mapping")
        legend_dims = list(color_mapping.dims) if color_mapping is not None else []

        if legend_kwargs is not False:
            legend_kwargs.setdefault("dim", legend_dims or ["color"])
        if legend_kwargs is not False and (legend is None or legend):
            plot_collection.add_legend(**legend_kwargs)

    if new_xlim is not None:
        plot_collection.map(
            set_xlim,
            "xlim",
            data=scalar_ds,
            ignore_aes="all",
            store_artist=False,
            limits=new_xlim,
        )

    colors_for_hover = None
    if point_rgba is not None and point_rgba.size:
        colors_for_hover = point_rgba.reshape(-1, point_rgba.shape[-1])

    if hover_label and n_data_points:
        labels_for_hover = [str(label) for label in flat_coord_labels]
        enable_hover_labels(
            backend,
            plot_collection,
            hover_format,
            labels_for_hover,
            colors_for_hover,
            y_flat,
        )
    return plot_collection
