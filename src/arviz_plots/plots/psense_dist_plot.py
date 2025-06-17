"""PsenseDist plot code."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.psense import power_scale_dataset
from xarray import Dataset, concat

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import process_group_variables_coords, set_grid_layout


def plot_psense_dist(
    dt,
    alphas=None,
    var_names=None,
    filter_vars=None,
    prior_var_names=None,
    likelihood_var_names=None,
    prior_coords=None,
    likelihood_coords=None,
    coords=None,
    sample_dims=None,
    kind=None,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "title",
            "rug",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "dist",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "title",
            "rug",
            "remove_axis",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[
        Literal["dist", "credible_interval", "point_estimate"], Mapping[str, Any] | Dataset
    ] = None,
    **pc_kwargs,
):
    """Plot power scaled posteriors.

    The posterior sensitivity is assessed by power-scaling the prior or likelihood and
    visualizing the resulting changes, using Pareto-smoothed importance sampling to
    avoid refitting as explained in [1]_.

    Parameters
    ----------
    dt : DataTree
        Input data
    alphas : tuple of float
        Lower and upper alpha values for power scaling. Defaults to (0.8, 1.25).
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    prior_var_names : str, optional
        Name of the log-prior variables to include in the power scaling sensitivity diagnostic
    likelihood_var_names : str, optional
        Name of the log-likelihood variables to include in the power scaling sensitivity diagnostic
    prior_coords : dict, optional
        Coordinates defining a subset over the group element for which to
        compute the log-prior sensitivity diagnostic
    likelihood_coords : dict, optional
        Coordinates defining a subset over the group element for which to
        compute the log-likelihood sensitivity diagnostic
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal distribution.
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to rcParam :data:`stats.point_estimate`
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to ``rcParams["stats.ci_prob"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals` except for "remove_axis"

    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * dist -> depending on the value of `kind` passed to:

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * credible_interval -> passed to :func:`~arviz_plots.visuals.line_x`
        * point_estimate -> passed to :func:`~arviz_plots.visuals.scatter_x`
        * point_estimate_text -> passed to :func:`~arviz_plots.visuals.point_estimate_text`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

    stats : mapping, optional
        Valid keys are:

        * dist -> passed to kde, ecdf, ...
        * credible_interval -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Select a single variable and generate a point-interval plot

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_psense_dist, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> rugby = load_arviz_data('rugby')
        >>> plot_psense_dist(rugby, var_names=["sd_att"], visuals={"kde":False})


    .. minigallery:: plot_psense_dist

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
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if stats is None:
        stats = {}
    else:
        stats = stats.copy()
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()

    if alphas is None:
        alphas = (0.8, 1.25)

    # Here we are generating new datasets for the prior and likelihood
    # by resampling the original dataset with the power scale weights
    # Instead we could have weighted KDEs/ecdfs/etc
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

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    color_cycle = pc_kwargs.get("color", plot_bknd.get_default_aes("color", 2, {}))
    if len(color_cycle) < 2:
        raise ValueError(
            f"Not enough values provided for color cycle, got {color_cycle} "
            "but at least 2 are needed"
        )

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("sharex", "row")
        pc_kwargs["figure_kwargs"].setdefault("sharey", "row")

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("color", [color_cycle[0], "black", color_cycle[1]])
        pc_kwargs.setdefault("y", [-0.05, -0.225, -0.4])
        pc_kwargs["aes"].setdefault("color", ["alpha"])
        pc_kwargs["aes"].setdefault("y", ["alpha"])
        pc_kwargs.setdefault("cols", ["component_group"])
        pc_kwargs.setdefault(
            "rows",
            ["__variable__"]
            + [
                dim
                for dim in distribution.dims
                if dim not in sample_dims + ["component_group", "alpha"]
            ],
        )

        pc_kwargs = set_grid_layout(pc_kwargs, plot_bknd, distribution, num_cols=2)
        plot_collection = PlotCollection.grid(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    visuals.setdefault("point_estimate_text", False)

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    aes_by_visuals.setdefault("point_estimate", ["color", "y"])
    aes_by_visuals.setdefault("credible_interval", ["color", "y"])

    if labeller is None:
        labeller = BaseLabeller()

    if kind == "hist":
        # Histograms are not great for overlapping distributions
        # But "step" histograms may be slightly easier to interpret than bars histograms
        # Using the same number of "bins" should help too
        visuals.setdefault("dist", {})
        visuals.setdefault("remove_axis", True)
        if visuals["dist"] is not False:
            visuals["dist"].setdefault("alpha", 0.3)
            visuals["dist"].setdefault("edgecolor", None)
            stats.setdefault("dist", {"density": True})

    if kind == "ecdf" and visuals.get("dist") is False:
        visuals.setdefault("remove_axis", True)

    plot_dist(
        distribution,
        var_names=None,
        filter_vars=None,
        group=None,
        coords=None,
        sample_dims=sample_dims,
        kind=kind,
        point_estimate=point_estimate,
        ci_kind=ci_kind,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats,
    )

    # Add legend for alpha parameter automatically
    plot_collection.add_legend(
        "alpha",
        title="Power Scale Factor",
    )

    return plot_collection
