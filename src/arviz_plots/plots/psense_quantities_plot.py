"""psense quantities plot code."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

from arviz_base import extract, rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.psense import power_scale_dataset
from xarray import concat

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_grid_layout,
)
from arviz_plots.visuals import hline, labelled_title, labelled_x, line_xy, scatter_xy, set_xticks


def plot_psense_quantities(
    dt,
    alphas=None,
    quantities=None,
    mcse=True,
    var_names=None,
    filter_vars=None,
    prior_var_names=None,
    likelihood_var_names=None,
    prior_coords=None,
    likelihood_coords=None,
    coords=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "prior_markers",
            "prior_lines",
            "likelihood_markers",
            "likelihood_lines",
            "mcse",
            "ticks",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "prior_markers",
            "prior_lines",
            "likelihood_markers",
            "likelihood_lines",
            "mcse",
            "ticks",
            "title",
            "legend",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    **pc_kwargs,
):
    """Plot power scaled posterior quantities.

    The posterior quantities are computed by power-scaling the prior or likelihood and
    visualizing the resulting changes, using Pareto-smoothed importance sampling to
    avoid refitting as explained in [1]_.

    Parameters
    ----------
    dt : DataTree
        Input data
    alphas : tuple of float
        Lower and upper alpha values for power scaling. Defaults to (0.8, 1.25).
    quantities : list of str
        Quantities to plot. Options are 'mean', 'sd', 'median'. For quantiles, use
        '0.25', '0.5', etc. Defaults to ['mean', 'sd'].
    mcse : bool
        Whether to plot the Monte Carlo standard error for each quantity. Defaults to True.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    prior_var_names : str, optional.
        Name of the log-prior variables to include in the power scaling sensitivity diagnostic
    likelihood_var_names : str, optional.
        Name of the log-likelihood variables to include in the power scaling sensitivity diagnostic
    prior_coords : dict, optional.
        Coordinates defining a subset over the group element for which to
        compute the log-prior sensitivity diagnostic
    likelihood_coords : dict, optional
        Coordinates defining a subset over the group element for which to
        compute the log-likelihood sensitivity diagnostic
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * prior_markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * prior_lines -> passed to :func:`~arviz_plots.visuals.line_xy`
        * likelihood_markers -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * likelihood_lines -> passed to :func:`~arviz_plots.visuals.line_xy`
        * mcse -> passed to :func:`~arviz_plots.visuals.hline`
        * ticks -> passed to :func:`~arviz_plots.visuals.set_xticks`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * legend -> passed to :class:`arviz_plots.PlotCollection.add_legend`

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Select a single parameter, one of the two likelihoods, and plot the mean, standard deviation,
    and 25th percentile.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_psense_quantities, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> rugby = load_arviz_data('rugby')
        >>> plot_psense_quantities(rugby,
        >>>                        var_names=["sd_att"],
        >>>                        likelihood_var_names=["home_points"],
        >>>                        quantities=["mean", "sd", "0.25"])


    .. minigallery:: plot_psense_quantities

    References
    ----------
    .. [1] Kallioinen et al, *Detecting and diagnosing prior and likelihood sensitivity with
        power-scaling*, Stat Comput 34, 57 (2024), https://doi.org/10.1007/s11222-023-10366-5

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if alphas is None:
        alphas = (0.8, 1.25)

    alphas_p1 = (alphas[0], 1, alphas[1])
    alphas_p1_labels = [str(val) for val in alphas_p1]

    if quantities is None:
        quantities = ["mean", "sd"]

    if isinstance(quantities, str):
        quantities = [quantities]

    if labeller is None:
        labeller = BaseLabeller()

    ds_posterior = extract(
        dt,
        var_names=var_names,
        filter_vars=filter_vars,
        group="posterior",
        combined=False,
        keep_dataset=True,
    )

    ds_prior = power_scale_dataset(
        dt,
        group="prior",
        alphas=alphas,
        sample_dims=sample_dims,
        group_var_names=prior_var_names,
        group_coords=prior_coords,
    )

    ds_likelihood = power_scale_dataset(
        dt,
        group="likelihood",
        alphas=alphas,
        sample_dims=sample_dims,
        group_var_names=likelihood_var_names,
        group_coords=likelihood_coords,
    )

    distribution = concat([ds_prior, ds_likelihood], dim="component_group").assign_coords(
        {"component_group": ["prior", "likelihood"]}
    )
    distribution = process_group_variables_coords(
        distribution, group=None, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    if len(sample_dims) > 1:
        # sample dims will have been stacked and renamed by `power_scale_dataset`
        sample_dims = ["sample"]

    to_concat_quantities = []
    to_concat_mcse = []
    name_quantities = []

    if "mean" in quantities:
        to_concat_quantities.append(distribution.azstats.mean(sample_dims))
        if mcse:
            to_concat_mcse.append(ds_posterior.azstats.mcse(method="mean"))
        name_quantities.append("mean")
    if "sd" in quantities:
        to_concat_quantities.append(distribution.std(sample_dims))
        if mcse:
            to_concat_mcse.append(ds_posterior.azstats.mcse(method="sd"))
        name_quantities.append("sd")
    if "median" in quantities:
        to_concat_quantities.append(distribution.azstats.median(sample_dims))
        if mcse:
            to_concat_mcse.append(ds_posterior.azstats.mcse(method="median"))
        name_quantities.append("median")
    for val in quantities:
        if val.replace(".", "").isnumeric():
            q = float(val)
            to_concat_quantities.append(
                distribution.quantile(q, sample_dims).rename_vars({"quantile": f"q={val}"})
            )
            if mcse:
                to_concat_mcse.append(ds_posterior.azstats.mcse(method="quantile", prob=q))
            name_quantities.append(f"q={val}")

    ds_quantities = concat(to_concat_quantities, "quantities").assign_coords(
        quantities=name_quantities
    )

    if mcse:
        mcse_quantities = concat(to_concat_mcse, "quantities").assign_coords(
            quantities=name_quantities
        )
        baseline_quantities = ds_quantities.sel(component_group="prior", alpha=1).drop_vars(
            ["alpha", "component_group"]
        )

        min_ = baseline_quantities - mcse_quantities * 2
        max_ = baseline_quantities + mcse_quantities * 2

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("color", ["component_group"])
        pc_kwargs.setdefault("cols", ["quantities"])
        pc_kwargs.setdefault(
            "rows",
            ["__variable__"]
            + [
                dim
                for dim in distribution.dims
                if dim not in sample_dims + ["component_group", "alpha"]
            ],
        )
        pc_kwargs = set_grid_layout(pc_kwargs, plot_bknd, ds_quantities)

        plot_collection = PlotCollection.grid(
            ds_quantities,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    aes_by_visuals.setdefault("quantities_marker", ["color", "marker"])
    aes_by_visuals.setdefault("quantities", ["color"])

    # plot quantities for prior-perturbations
    ## markers
    prior_ms_kwargs = get_visual_kwargs(visuals, "prior_markers")

    if prior_ms_kwargs is not False:
        _, _, prior_ms_ignore = filter_aes(
            plot_collection, aes_by_visuals, "prior_markers", sample_dims
        )
        prior_ms_kwargs.setdefault("marker", "C0")
        prior_ms_kwargs.setdefault("color", "C0")

        plot_collection.map(
            scatter_xy,
            "prior_markers",
            data=ds_quantities.sel(component_group="prior"),
            x=ds_quantities.alpha,
            ignore_aes=prior_ms_ignore,
            **prior_ms_kwargs,
        )
    ## lines
    prior_ls_kwargs = get_visual_kwargs(visuals, "prior_lines")

    if prior_ls_kwargs is not False:
        _, _, prior_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "prior_lines", sample_dims
        )
        prior_ls_kwargs.setdefault("color", "C0")

        plot_collection.map(
            line_xy,
            "prior_lines",
            data=ds_quantities.sel(component_group="prior"),
            x=ds_quantities.alpha,
            ignore_aes=prior_ls_ignore,
            **prior_ls_kwargs,
        )

    # plot quantities for likelihood-perturbations
    ## markers
    likelihood_ms_kwargs = get_visual_kwargs(visuals, "likelihood_markers")

    if likelihood_ms_kwargs is not False:
        _, _, likelihood_ms_ignore = filter_aes(
            plot_collection, aes_by_visuals, "likelihood_markers", sample_dims
        )

        likelihood_ms_kwargs.setdefault("marker", "C5")
        likelihood_ms_kwargs.setdefault("color", "C1")

        plot_collection.map(
            scatter_xy,
            "likelihood_markers",
            data=ds_quantities.sel(component_group="likelihood"),
            x=ds_quantities.alpha,
            ignore_aes=likelihood_ms_ignore,
            **likelihood_ms_kwargs,
        )
    ## lines
    likelihood_ls_kwargs = get_visual_kwargs(visuals, "likelihood_lines")

    if likelihood_ls_kwargs is not False:
        _, _, likelihood_ls_ignore = filter_aes(
            plot_collection, aes_by_visuals, "likelihood_lines", sample_dims
        )

        likelihood_ls_kwargs.setdefault("color", "C1")

        plot_collection.map(
            line_xy,
            "likelihood_lines",
            data=ds_quantities.sel(component_group="likelihood"),
            x=ds_quantities.alpha,
            ignore_aes=likelihood_ls_ignore,
            **likelihood_ls_kwargs,
        )

    # plot mcse
    if mcse:
        mcse_kwargs = get_visual_kwargs(visuals, "mcse")
        _, _, mcse_ignore = filter_aes(plot_collection, aes_by_visuals, "mcse", sample_dims)
        if mcse_kwargs is not False:
            mcse_kwargs.setdefault("color", "B2")
            mcse_kwargs.setdefault("linestyle", "C1")

            plot_collection.map(hline, "mcse", data=min_, ignore_aes=mcse_ignore, **mcse_kwargs)
            plot_collection.map(hline, "mcse", data=max_, ignore_aes=mcse_ignore, **mcse_kwargs)

    # set ticks
    ticks_kwargs = get_visual_kwargs(visuals, "ticks")
    _, _, ticks_ignore = filter_aes(plot_collection, aes_by_visuals, "ticks", sample_dims)

    if ticks_kwargs is not False:
        plot_collection.map(
            set_xticks,
            "ticks",
            values=alphas_p1,
            labels=alphas_p1_labels,
            ignore_aes=ticks_ignore,
            store_artist=backend == "none",
            **ticks_kwargs,
        )

    # set xlabel
    _, xlabels_aes, xlabels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "xlabel", sample_dims
    )
    xlabel_kwargs = visuals.get("xlabel", {}).copy()
    if xlabel_kwargs is not False:
        if "color" not in xlabels_aes:
            xlabel_kwargs.setdefault("color", "B1")

        xlabel_kwargs.setdefault("text", "Power-scaling α")

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=xlabels_ignore,
            subset_info=True,
            **xlabel_kwargs,
        )

    # title
    title_kwargs = get_visual_kwargs(visuals, "title")
    _, _, title_ignore = filter_aes(plot_collection, aes_by_visuals, "title", sample_dims)

    if title_kwargs is not False:
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    # legend
    legend_kwargs = get_visual_kwargs(visuals, "legend")
    if legend_kwargs is not False:
        legend_kwargs.setdefault("dim", ["component_group"])

        plot_collection.add_legend(**legend_kwargs)

    return plot_collection
