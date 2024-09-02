"""ess plot code."""

from copy import copy
from importlib import import_module

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection, process_facet_dims
from arviz_plots.plots.utils import filter_aes, get_group, process_group_variables_coords
from arviz_plots.visuals import (
    annotate_xy,
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
    trace_rug,
)


def plot_ess(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind="local",
    relative=False,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    extra_methods=False,
    min_ess=400,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot effective sample size plots.

    Parameters
    ----------
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.
    var_names : str or sequence of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default None
        If None, interpret `var_names` as the real variables names.
        If “like”, interpret `var_names` as substrings of the real variables names.
        If “regex”, interpret `var_names` as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"local", "quantile"}, default "local"
        Specify the kind of plot:

        * The ``kind="local"`` argument generates the ESS' local efficiency
          for estimating small-interval probability of a desired posterior.
        * The ``kind="quantile"`` argument generates the ESS' local efficiency
          for estimating quantiles of a desired posterior.

    relative : bool, default False
        Show relative ess in plot ``ress = ess / N``.
    rug : bool, default False
        Add a `rug plot <https://en.wikipedia.org/wiki/Rug_plot>`_ for a specific subset of values.
    rug_kind : str, default "diverging"
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points : int, default 20
        Number of points for which to plot their quantile/local ess or number of subsets
        in the evolution plot.
    extra_methods : bool, default False
        Plot mean and sd ESS as horizontal lines.
    min_ess : int, default 400
        Minimum number of ESS desired. If ``relative=True`` the line is plotted at
        ``min_ess / n_samples`` for local and quantile kinds
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str or False}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * ess -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * rug -> passed to :func:`~.visuals.trace_rug`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * mean -> passed to :func:`~arviz.plots.visuals.line_xy`
        * mean_text -> passed to :func:`~arviz.plots.visuals.annotate_xy`
        * sd_text -> passed to :func:`~arviz.plots.visuals.annotate_xy`
        * sd -> passed to :func:`~arviz.plots.visuals.line_xy`
        * min_ess -> passed to :func:`~arviz.plots.visuals.line_xy`

    stats_kwargs : mapping, optional
        Valid keys are:

        * ess -> passed to ess, method = 'local' or 'quantile' based on `kind`
        * mean -> passed to ess, method='mean'
        * sd -> passed to ess, method='sd'

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    Notes
    -----
    Depending on the number of models, a slight x-axis separation aesthetic is applied for each
    ess point for distinguishability in case of overlap

    See Also
    --------
    :ref:`plots_intro` :
        General introduction to batteries-included plotting functions, common use and logic overview

    Examples
    --------
    We can manually map the color to the variable, and have the mapping apply
    to the title too instead of only the ess markers:

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ess, style
        >>> style.use("arviz-clean")
        >>> from arviz_base import load_arviz_data
        >>> non_centered = load_arviz_data('non_centered_eight')
        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     pc_kwargs={"aes": {"color": ["__variable__"]}},
        >>>     aes_map={"title": ["color"]},
        >>> )

    We can add extra methods to plot the mean and standard deviation as lines, and adjust
    the minimum ess baseline as well:

    .. plot::
        :context: close-figs

        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     extra_methods=True,
        >>>     min_ess=200,
        >>> )

    Rugs can also be added:

    .. plot::
        :context: close-figs

        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     rug=True,
        >>> )

    Relative ESS can be plotted instead of absolute:

    .. plot::
        :context: close-figs

        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     relative=True,
        >>> )

    We can also adjust the number of points:

    .. plot::
        :context: close-figs

        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     n_points=10,
        >>> )

    .. minigallery:: plot_ess

    """
    # initial defaults
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    ylabel = "{}"

    # mutable inputs
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if stats_kwargs is None:
        stats_kwargs = {}

    # processing dt/group/coords/filtering
    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    # ensuring plot_kwargs['rug'] is not False
    rug_kwargs = copy(plot_kwargs.get("rug", {}))
    if rug_kwargs is False:
        raise ValueError("plot_kwargs['rug'] can't be False, use rug=False to remove the rug")

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    # set plot collection initialization defaults
    if plot_collection is None:
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]
            + [dim for dim in distribution.dims if dim not in {"model"}.union(sample_dims)],
        )
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        if "model" in distribution:
            pc_kwargs["aes"].setdefault("color", ["model"])
            n_models = distribution.sizes["model"]
            x_diff = min(1 / n_points / 3, 1 / n_points * n_models / 10)
            pc_kwargs.setdefault("x", np.linspace(-x_diff, x_diff, n_models))
            pc_kwargs["aes"].setdefault("x", ["model"])
        aux_dim_list = [dim for dim in pc_kwargs["cols"] if dim != "__variable__"]
        figsize = pc_kwargs.get("plot_grid_kws", {}).get("figsize", None)
        figsize_units = pc_kwargs.get("plot_grid_kws", {}).get("figsize_units", "inches")
        if figsize is None:
            coeff = 0.2
            if "chain" in distribution.dims:
                coeff += 0.1
            if "model" in distribution.dims:
                coeff += 0.1 * distribution.sizes["model"]
            n_plots, _ = process_facet_dims(distribution, pc_kwargs["cols"])
            col_wrap = pc_kwargs["col_wrap"]
            print(f"\n n_plots = {n_plots},\n col_wrap = {col_wrap}")
            if n_plots <= col_wrap:
                n_rows, n_cols = 1, n_plots
            else:
                div_mod = divmod(n_plots, col_wrap)
                n_rows = div_mod[0] + (div_mod[1] != 0)
                n_cols = col_wrap
            print(f"\n n_rows = {n_rows},\n n_cols = {n_cols}")
            figsize = plot_bknd.scale_fig_size(
                figsize,
                rows=n_rows,
                cols=n_cols,
                figsize_units=figsize_units,
            )
            print(f"\n figsize = {figsize!r}")
            figsize_units = "dots"
        pc_kwargs["plot_grid_kws"]["figsize"] = figsize
        pc_kwargs["plot_grid_kws"]["figsize_units"] = figsize_units
        plot_collection = PlotCollection.grid(
            distribution,
            backend=backend,
            **pc_kwargs,
        )
    else:
        aux_dim_list = list(
            set(
                dim for child in plot_collection.viz.children.values() for dim in child["plot"].dims
            )
        )

    # set plot collection dependent defaults (like aesthetics mappings for each artist)
    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault(kind, plot_collection.aes_set.difference({"overlay"}))
    aes_map.setdefault("rug", {"overlay"})
    if "model" in distribution:
        aes_map.setdefault("mean", {"color"})
        aes_map.setdefault("sd", {"color"})
        aes_map.setdefault("min_ess", {"color"})
    if "mean" in aes_map and "mean_text" not in aes_map:
        aes_map["mean_text"] = aes_map["mean"]
    if "sd" in aes_map and "sd_text" not in aes_map:
        aes_map["sd_text"] = aes_map["sd"]
    if labeller is None:
        labeller = BaseLabeller()

    # compute and add ess subplots
    ess_kwargs = copy(plot_kwargs.get("ess", {}))

    if ess_kwargs is not False:
        ess_dims, _, ess_ignore = filter_aes(plot_collection, aes_map, kind, sample_dims)
        if kind == "local":
            probs = np.linspace(0, 1, n_points, endpoint=False)
            ylabel = "{} for small intervals"
        elif kind == "quantile":
            probs = np.linspace(1 / n_points, 1 - 1 / n_points, n_points)
            ylabel = "{} for quantiles"
        xdata = probs

        ess_y_dataset = xr.concat(
            [
                distribution.azstats.ess(
                    dims=ess_dims,
                    method=kind,
                    relative=relative,
                    prob=[p, (p + 1 / n_points)] if kind == "local" else p,
                    **stats_kwargs.get("ess", {}),
                )
                for p in probs
            ],
            dim="ess_dim",
        )

        xdata_da = xr.DataArray(xdata, dims="ess_dim")
        # broadcasting xdata_da to match shape of each variable in ess_y_dataset and
        # creating a new dataset from dict of broadcasted xdata
        xdata_dataset = xr.Dataset(
            {var_name: xdata_da.broadcast_like(da) for var_name, da in ess_y_dataset.items()}
        )
        # concatenating xdata_dataset and ess_y_dataset along plot_axis
        ess_dataset = xr.concat([xdata_dataset, ess_y_dataset], dim="plot_axis").assign_coords(
            plot_axis=["x", "y"]
        )

        plot_collection.map(
            scatter_xy, "ess", data=ess_dataset, ignore_aes=ess_ignore, **ess_kwargs
        )

    # plot rug
    # overlaying divergences(or other 'rug_kind') for each chain
    if rug:
        sample_stats = get_group(dt, "sample_stats", allow_missing=True)
        if (
            sample_stats is not None
            and rug_kind in sample_stats.data_vars
            and np.any(sample_stats[rug_kind])  # 'diverging' by default
            and rug_kwargs is not False
        ):
            rug_mask = dt.sample_stats[rug_kind]  # 'diverging' by default
            _, div_aes, div_ignore = filter_aes(plot_collection, aes_map, "rug", sample_dims)
            if "color" not in div_aes:
                rug_kwargs.setdefault("color", "black")
            if "marker" not in div_aes:
                rug_kwargs.setdefault("marker", "|")
            if "size" not in div_aes:
                rug_kwargs.setdefault("size", 30)
            div_reduce_dims = [dim for dim in distribution.dims if dim not in aux_dim_list]

            values = distribution.azstats.compute_ranks(relative=True)

            plot_collection.map(
                trace_rug,
                "rug",
                data=values,
                ignore_aes=div_ignore,
                y=distribution.min(div_reduce_dims),
                mask=rug_mask,
                xname=False,
                **rug_kwargs,
            )  # note: after plot_ppc merge, the `trace_rug` function might change

    # defining x_range (used for mean, sd, minimum ess plotting)
    x_range = [0, 1]
    x_range = xr.DataArray(x_range)

    # getting backend specific linestyles
    linestyles = plot_bknd.get_default_aes("linestyle", 4, {})
    # and default color
    default_color = plot_bknd.get_default_aes("color", 1, {})[0]

    # plot mean and sd and annotate them
    if extra_methods is not False:
        # computing mean_ess
        mean_dims, mean_aes, mean_ignore = filter_aes(plot_collection, aes_map, "mean", sample_dims)
        mean_ess = distribution.azstats.ess(
            dims=mean_dims, method="mean", relative=relative, **stats_kwargs.get("mean", {})
        )

        # computing sd_ess
        sd_dims, sd_aes, sd_ignore = filter_aes(plot_collection, aes_map, "sd", sample_dims)
        sd_ess = distribution.azstats.ess(
            dims=sd_dims, method="sd", relative=relative, **stats_kwargs.get("sd", {})
        )

        mean_kwargs = copy(plot_kwargs.get("mean", {}))
        if mean_kwargs is not False:
            # getting 2nd default linestyle for chosen backend and assigning it by default
            mean_kwargs.setdefault("linestyle", linestyles[1])

            if "color" not in mean_aes:
                mean_kwargs.setdefault("color", default_color)

            plot_collection.map(
                line_xy,
                "mean",
                data=mean_ess,
                x=x_range,
                ignore_aes=mean_ignore,
                **mean_kwargs,
            )

        sd_kwargs = copy(plot_kwargs.get("sd", {}))
        if sd_kwargs is not False:
            sd_kwargs.setdefault("linestyle", linestyles[2])

            if "color" not in sd_aes:
                sd_kwargs.setdefault("color", default_color)

            plot_collection.map(
                line_xy, "sd", data=sd_ess, ignore_aes=sd_ignore, x=x_range, **sd_kwargs
            )

        sd_va_align = None
        mean_va_align = None
        if mean_ess is not None and sd_ess is not None:
            sd_va_align = xr.where(mean_ess < sd_ess, "bottom", "top")
            mean_va_align = xr.where(mean_ess < sd_ess, "top", "bottom")

        mean_text_kwargs = copy(plot_kwargs.get("mean_text", {}))
        if (
            mean_text_kwargs is not False and mean_ess is not None
        ):  # mean_ess has to exist for an annotation to be applied
            _, mean_text_aes, mean_text_ignore = filter_aes(
                plot_collection, aes_map, "mean_text", sample_dims
            )

            if "color" not in mean_text_aes:
                mean_text_kwargs.setdefault("color", "black")

            mean_text_kwargs.setdefault("x", 1)
            mean_text_kwargs.setdefault("horizontal_align", "right")

            # pass the mean vertical_align data for vertical alignment setting
            if mean_va_align is not None:
                vertical_align = mean_va_align
            else:
                vertical_align = "bottom"
            mean_text_kwargs.setdefault("vertical_align", vertical_align)

            plot_collection.map(
                annotate_xy,
                "mean_text",
                text="mean",
                data=mean_ess,
                ignore_aes=mean_text_ignore,
                **mean_text_kwargs,
            )

        sd_text_kwargs = copy(plot_kwargs.get("sd_text", {}))
        if (
            sd_text_kwargs is not False and sd_ess is not None
        ):  # sd_ess has to exist for an annotation to be applied
            _, sd_text_aes, sd_text_ignore = filter_aes(
                plot_collection, aes_map, "sd_text", sample_dims
            )

            if "color" not in sd_text_aes:
                sd_text_kwargs.setdefault("color", "black")

            sd_text_kwargs.setdefault("x", 1)
            sd_text_kwargs.setdefault("horizontal_align", "right")

            # pass the sd vertical_align data for vertical alignment setting
            if sd_va_align is not None:
                vertical_align = sd_va_align
            else:
                vertical_align = "top"
            sd_text_kwargs.setdefault("vertical_align", vertical_align)

            plot_collection.map(
                annotate_xy,
                "sd_text",
                text="sd",
                data=sd_ess,
                ignore_aes=sd_text_ignore,
                **sd_text_kwargs,
            )

    # plot minimum ess
    min_ess_kwargs = copy(plot_kwargs.get("min_ess", {}))

    if min_ess_kwargs is not False:
        _, min_ess_aes, min_ess_ignore = filter_aes(
            plot_collection, aes_map, "min_ess", sample_dims
        )

        if relative:
            min_ess = min_ess / n_points

        min_ess_kwargs.setdefault("linestyle", linestyles[3])

        if "color" not in min_ess_aes:
            min_ess_kwargs.setdefault("color", "gray")

        plot_collection.map(
            line_xy,
            "min_ess",
            data=distribution,
            ignore_aes=min_ess_ignore,
            x=x_range,
            y=min_ess,
            **min_ess_kwargs,
        )

    # plot titles for each facetted subplot
    title_kwargs = copy(plot_kwargs.get("title", {}))

    if title_kwargs is not False:
        _, title_aes, title_ignore = filter_aes(plot_collection, aes_map, "title", sample_dims)
        if "color" not in title_aes:
            title_kwargs.setdefault("color", "black")
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    # plot x and y axis labels
    # Add varnames as x and y labels
    _, labels_aes, labels_ignore = filter_aes(plot_collection, aes_map, "xlabel", sample_dims)
    xlabel_kwargs = plot_kwargs.get("xlabel", {}).copy()
    if xlabel_kwargs is not False:
        if "color" not in labels_aes:
            xlabel_kwargs.setdefault("color", "black")

        # formatting ylabel and setting xlabel
        xlabel_kwargs.setdefault("text", "Quantile")

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=labels_ignore,
            subset_info=True,
            **xlabel_kwargs,
        )

    _, labels_aes, labels_ignore = filter_aes(plot_collection, aes_map, "ylabel", sample_dims)
    ylabel_kwargs = plot_kwargs.get("ylabel", {}).copy()
    if ylabel_kwargs is not False:
        if "color" not in labels_aes:
            ylabel_kwargs.setdefault("color", "black")

        if relative is not False:
            ylabel_text = ylabel.format("Relative ESS")
        else:
            ylabel_text = ylabel.format("ESS")
        ylabel_kwargs.setdefault("text", ylabel_text)

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=labels_ignore,
            subset_info=True,
            **ylabel_kwargs,
        )

    return plot_collection
