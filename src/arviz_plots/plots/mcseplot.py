"""mcse plot code."""

# imports
# import warnings
from copy import copy
from importlib import import_module

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, get_group, process_group_variables_coords
from arviz_plots.visuals import (
    annotate_xy,
    error_bar,
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
    trace_rug,
)

# from arviz_base.sel_utils import xarray_sel_iter


# from arviz_stats.numba import array_stats


def plot_mcse(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    errorbar=False,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    extra_methods=False,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot quantile  Monte Carlo Standard Error.

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
    errorbar : bool, default False
        Plot quantile value +/- mcse instead of plotting mcse.
    rug : bool, default False
        Add a `rug plot <https://en.wikipedia.org/wiki/Rug_plot>`_ for a specific subset of values.
    rug_kind : str, default "diverging"
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points : int, default 20
        Number of points for which to plot their quantile MCSE.
    extra_methods : bool, default False
        Plot mean and sd MCSE as horizontal lines. Only taken into account when
        ``errorbar=False``.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str or False}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * mcse -> passed to :func:`~arviz_plots.visuals.scatter_xy`
            if ``errorbar=False``.
            Passed to :func:`~arviz_plots.visuals.errorbar`
            if ``errorbar=True``.

        * rug -> passed to :func:`~.visuals.trace_rug`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * mean -> passed to :func:`~arviz.plots.visuals.line_xy`
        * sd -> passed to :func:`~arviz.plots.visuals.line_xy`
        * mean_text -> passed to :func:`~arviz.plots.visuals.annotate_xy`
        * sd_text -> passed to :func:`~arviz.plots.visuals.annotate_xy`

    stats_kwargs : mapping, optional
        Valid keys are:

        * mcse -> passed to mcse, method = 'quantile'
        * mean -> passed to mcse, method='mean'
        * sd -> passed to mcse, method='sd'

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection


    """
    # initial defaults
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

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

    # set plot collection initialization defaults if it doesnt exist
    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("col_wrap", 5)
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]
            + [dim for dim in distribution.dims if dim not in {"model"}.union(sample_dims)],
        )
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        if "model" in distribution:
            pc_kwargs["aes"].setdefault("color", ["model"])
            n_models = distribution.sizes["model"]
            x_diff = min(1 / n_points / 3, 1 / n_points * n_models / 10)
            pc_kwargs.setdefault("x", np.linspace(-x_diff, x_diff, n_models))
            pc_kwargs["aes"].setdefault("x", ["model"])
        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    # set plot collection dependent defaults (like aesthetics mappings for each artist)
    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()
    aes_map.setdefault("mcse", plot_collection.aes_set.difference({"overlay"}))
    aes_map.setdefault("rug", {"overlay"})
    if "model" in distribution:
        aes_map.setdefault("mean", {"color"})
        aes_map.setdefault("sd", {"color"})
    if "mean" in aes_map and "mean_text" not in aes_map:
        aes_map["mean_text"] = aes_map["mean"]
    if "sd" in aes_map and "sd_text" not in aes_map:
        aes_map["sd_text"] = aes_map["sd"]
    if labeller is None:
        labeller = BaseLabeller()

    # compute and add mcse subplots
    mcse_kwargs = copy(plot_kwargs.get("mcse", {}))

    if mcse_kwargs is not False:
        mcse_dims, _, mcse_ignore = filter_aes(plot_collection, aes_map, "mcse", sample_dims)
        probs = np.linspace(1 / n_points, 1 - 1 / n_points, n_points)
        xdata = probs
        print(f"\n mcse_dims = {mcse_dims}")

        mcse_y_dataset = xr.concat(
            [
                distribution.azstats.mcse(
                    dims=mcse_dims,
                    method="quantile",
                    prob=p,
                    **stats_kwargs.get("mcse", {}),
                )
                for p in probs
            ],
            dim="mcse_dim",
        )
        # print(f"\n mcse_y_dataset = {mcse_y_dataset}")

        xdata_da = xr.DataArray(xdata, dims="mcse_dim")
        # broadcasting xdata_da to match shape of each variable in mcse_y_dataset and
        # creating a new dataset from dict of broadcasted xdata
        xdata_dataset = xr.Dataset(
            {var_name: xdata_da.broadcast_like(da) for var_name, da in mcse_y_dataset.items()}
        )
        # concatenating xdata_dataset and ess_y_dataset along plot_axis
        mcse_dataset = xr.concat([xdata_dataset, mcse_y_dataset], dim="plot_axis").assign_coords(
            plot_axis=["x", "y"]
        )

        print(f"\n mcse_dataset = {mcse_dataset!r}")
        # print(f"\n distribution = {distribution!r}")

        if errorbar is False:
            plot_collection.map(
                scatter_xy, "mcse", data=mcse_dataset, ignore_aes=mcse_ignore, **mcse_kwargs
            )

        # else:
        # use new errorbar visual element function to plot errorbars
        else:
            # print(f"\n final z_mcse_dataset = {z_mcse_dataset}")
            quantiles_dataset = distribution.quantile(probs, dim=mcse_dims)
            print(f"\n quantiles_dataset = {quantiles_dataset}")

            # for now the quantile_values can be computed in the visual element function itself
            plot_collection.map(
                error_bar,
                "mcse",
                data=mcse_dataset,
                ignore_aes=mcse_ignore,
                quantiles_dataset=quantiles_dataset,
                # distribution=distribution,  # map() subsets this before passing to error_bar
                **mcse_kwargs,
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

            values = distribution.azstats.compute_ranks(relative=True)

            plot_collection.map(
                trace_rug,
                "rug",
                data=values,
                ignore_aes=div_ignore,
                y=0,
                mask=rug_mask,
                xname=False,
                **rug_kwargs,
            )  # note: after plot_ppc merge, the `trace_rug` function might change

    # defining x_range (used for mean, sd plotting)
    x_range = [0, 1]
    x_range = xr.DataArray(x_range)

    # getting backend specific linestyles
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    linestyles = plot_bknd.get_default_aes("linestyle", 4, {})
    # and default color
    default_color = plot_bknd.get_default_aes("color", 1, {})[0]

    # plot mean and sd
    if extra_methods is not False:
        if errorbar is not False:
            raise ValueError("Please ensure errorbar=False if you want to plot mean and sd")

        # computing mean_mcse
        mean_dims, mean_aes, mean_ignore = filter_aes(plot_collection, aes_map, "mean", sample_dims)
        mean_mcse = distribution.azstats.mcse(
            dims=mean_dims, method="mean", **stats_kwargs.get("mean", {})
        )

        # computing sd_mcse
        sd_dims, sd_aes, sd_ignore = filter_aes(plot_collection, aes_map, "sd", sample_dims)
        sd_mcse = distribution.azstats.mcse(dims=sd_dims, method="sd", **stats_kwargs.get("sd", {}))

        mean_kwargs = copy(plot_kwargs.get("mean", {}))
        if mean_kwargs is not False:
            # getting 2nd default linestyle for chosen backend and assigning it by default
            mean_kwargs.setdefault("linestyle", linestyles[1])

            if "color" not in mean_aes:
                mean_kwargs.setdefault("color", default_color)

            plot_collection.map(
                line_xy,
                "mean",
                data=mean_mcse,
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
                line_xy, "sd", data=sd_mcse, ignore_aes=sd_ignore, x=x_range, **sd_kwargs
            )

        sd_va_align = None
        mean_va_align = None
        if mean_mcse is not None and sd_mcse is not None:
            sd_va_align = xr.where(mean_mcse < sd_mcse, "bottom", "top")
            mean_va_align = xr.where(mean_mcse < sd_mcse, "top", "bottom")

        mean_text_kwargs = copy(plot_kwargs.get("mean_text", {}))
        if (
            mean_text_kwargs is not False and mean_mcse is not None
        ):  # mean_mcse has to exist for an annotation to be applied
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

            plot_collection.map(
                annotate_xy,
                "mean_text",
                text="mean",
                data=mean_mcse,
                vertical_align=vertical_align,
                ignore_aes=mean_text_ignore,
                **mean_text_kwargs,
            )

        sd_text_kwargs = copy(plot_kwargs.get("sd_text", {}))
        if (
            sd_text_kwargs is not False and sd_mcse is not None
        ):  # sd_mcse has to exist for an annotation to be applied
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

            plot_collection.map(
                annotate_xy,
                "sd_text",
                text="sd",
                data=sd_mcse,
                vertical_align=vertical_align,
                ignore_aes=sd_text_ignore,
                **sd_text_kwargs,
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

        ylabel_text = r"Value $\pm$ MCSE for quantiles" if errorbar else "MCSE for quantiles"

        ylabel_kwargs.setdefault("text", ylabel_text)

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=labels_ignore,
            subset_info=True,
            **ylabel_kwargs,
        )

    print(f"\n pc.viz = {plot_collection.viz}")

    print(f"\n pc.aes = {plot_collection.aes}")

    return plot_collection
