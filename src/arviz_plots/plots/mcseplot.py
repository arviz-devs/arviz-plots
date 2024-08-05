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
        aux_dim_list = [dim for dim in pc_kwargs["cols"] if dim != "__variable__"]
        plot_collection = PlotCollection.wrap(
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
    aes_map.setdefault("mcse", plot_collection.aes_set.difference({"overlay"}))
    aes_map.setdefault("rug", {"overlay"})
    if "model" in distribution:
        aes_map.setdefault("mean", {"color"})
        aes_map.setdefault("sd", {"color"})
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
            # ----------------------------------------------
            # ||                WIP                       ||
            # ----------------------------------------------
            # data to pass to .map() and get facetted:
            # - mcse_dataset

            # data that has to be computed before: quantile values
            # thi can be computed with distribution and probs and the quantile func from arviz-stats

            # ^for every subsel of variable/dimension-by-coords in distribution, the values are
            # flattened. then call quantile(values, probs) for each of these subselections, and
            # concatenate it to the mcse_dataset for later facetting by map() and destructuring by
            # the errorbar visual element func

            a = """
            plotters = xarray_sel_iter(
                distribution, skip_dims=mcse_dims
            )  # create subselections of distribution, looping over all dims except mcse_dims

            z_mcse_dataset = (
                xr.Dataset()
            )  # to be concatenated with mcse_dataset later along dim=plot_axis
            # z_mcse_dict = {}

            # compute quantile values for each subselection of distribution
            for var_name, sel, isel in plotters:
                print(f"\nvar={var_name} | sel={sel} | sel.items()={sel.items()} | isel={isel}")
                da = distribution[var_name].sel(sel)
                da = da.values.flatten()
                # print(f"\n da = {da}")
                quantile_values = array_stats.quantile(da, probs)
                # print(f"quantile_values = {quantile_values}")

                # Convert quantile_values to a DataArray
                quantile_da = xr.DataArray(quantile_values, dims=["mcse_dim"], name=var_name)
                print(f"\n quantile_da = {quantile_da}")

                # Expand dims of dataarray and assign new coords 'z' and sel dim coords
                quantile_da_expanded = quantile_da.expand_dims({"plot_axis": ["z"]})
                # .assign_coords(
                #    {"plot_axis": ["z"]}
                # )
                # .to_dataset(  # .assign_coords(plot_axis=["z"])
                #    name=var_name
                # )
                if sel:  # adding sel dims to quantile_da
                    sel_expanded = {key: [value] for key, value in sel.items()}
                    print(f"sel_expanded = {sel_expanded}")
                    quantile_da_expanded = quantile_da_expanded.expand_dims(sel_expanded)
                    #    dim=list(sel.keys())
                    # ).assign_coords(**sel)
                # quantile_da_expanded = quantile_da_expanded.assign_coords(**sel)

                print(f"\n quantile_da_expanded = {quantile_da_expanded}")

                # print(f"\n mcse_dataset sel = {mcse_dataset[var_name].sel(**sel)!r}")
                # print(f"\n mcse_dataset isel = {mcse_dataset[var_name].isel(isel)!r}")

                if var_name in z_mcse_dataset:
                    combined_dataarray = z_mcse_dataset[var_name].combine_first(
                        quantile_da_expanded
                    )
                    print(f"\n combined_dataarray = {combined_dataarray}")
                    z_mcse_dataset.update({var_name: (var_name, combined_dataarray)})
                    # .concat(), .merge() are not in-place functions. .update() is in-place but
                    # not working rn

                    # z_mcse_dataset[var_name] =
                    # xr.concat(
                    #    [z_mcse_dataset[var_name], quantile_da_expanded],
                    #    dim="school",
                    # )
                else:
                    z_mcse_dataset[var_name] = quantile_da_expanded
                    # print(f"\n adding var {var_name} to z_mcse_dataset: {z_mcse_dataset}")

            print(f"\n z_mcse_dataset = {z_mcse_dataset}")"""
            print(len(a))

            # print(f"\n final z_mcse_dataset = {z_mcse_dataset}")

            # for now the quantile_values can be computed in the visual element function itself
            plot_collection.map(
                error_bar,
                "mcse",
                data=mcse_dataset,
                ignore_aes=mcse_ignore,
                distribution=distribution,  # map() subsets this before passing to error_bar
                **mcse_kwargs,
            )

    # WIP: plot rug. rankdata func to be used.
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
            # WIP: if using a default linewidth once defined in backend/agnostic defaults
            # if "width" not in div_aes:
            #    # get default linewidth for backends
            #    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
            #    default_linewidth = plot_bknd.get_default_aes("linewidth", 1, {})
            #    rug_kwargs.setdefault("width", default_linewidth)
            if "size" not in div_aes:
                rug_kwargs.setdefault("size", 30)
            div_reduce_dims = [dim for dim in distribution.dims if dim not in aux_dim_list]

            # xname is used to pick subset of dataset in map() to be masked
            xname = None
            default_xname = sample_dims[0] if len(sample_dims) == 1 else "draw"
            if (default_xname not in distribution.dims) or (
                not np.issubdtype(distribution[default_xname].dtype, np.number)
            ):
                default_xname = None
            xname = rug_kwargs.get("xname", default_xname)
            rug_kwargs["xname"] = xname

            draw_length = (
                distribution.sizes[sample_dims[0]]
                if len(sample_dims) == 1
                else distribution.sizes["draw"]
            )  # used to scale xvalues to between 0-1

            plot_collection.map(
                trace_rug,
                "rug",
                data=distribution,
                ignore_aes=div_ignore,
                # xname=xname,
                y=distribution.min(div_reduce_dims),
                mask=rug_mask,
                scale=draw_length,
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
        mean_kwargs = copy(plot_kwargs.get("mean", {}))
        if mean_kwargs is not False:
            mean_dims, mean_aes, mean_ignore = filter_aes(
                plot_collection, aes_map, "mean", sample_dims
            )
            mean_mcse = distribution.azstats.mcse(
                dims=mean_dims, method="mean", **stats_kwargs.get("mean", {})
            )
            # print(f"\n mean_mcse = {mean_mcse}")

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
            sd_dims, sd_aes, sd_ignore = filter_aes(plot_collection, aes_map, "sd", sample_dims)
            sd_mcse = distribution.azstats.mcse(
                dims=sd_dims, method="sd", **stats_kwargs.get("sd", {})
            )
            # print(f"\n sd_mcse = {sd_mcse}")
            sd_kwargs.setdefault("linestyle", linestyles[2])

            if "color" not in sd_aes:
                sd_kwargs.setdefault("color", default_color)

            plot_collection.map(
                line_xy, "sd", data=sd_mcse, ignore_aes=sd_ignore, x=x_range, **sd_kwargs
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
            store_artist=False,
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
            store_artist=False,
            **ylabel_kwargs,
        )

    return plot_collection
