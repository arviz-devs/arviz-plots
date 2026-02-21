"""Posterior predictive check for survival/censored data using Kaplan-Meier curves."""

from importlib import import_module
from typing import Any, Literal, Mapping, Sequence

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.survival import generate_survival_curves, kaplan_meier

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import ecdf_line, labelled_title, labelled_x, labelled_y


def plot_ppc_censored(
    dt,
    *,
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    num_samples=100,
    extrapolation_factor=1.2,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "observed_km",
            "predictive",
            "xlabel",
            "ylabel",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "observed_km",
            "predictive",
            "xlabel",
            "ylabel",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    **pc_kwargs,
):
    """Plot Kaplan-Meier survival curve [1]_ vs predictive draws.

    Instead of plotting the raw data observation and predictions, as is common in posterior
    predictive checks, this function computes the Kaplan-Meier survival curves for observed
    and for predictive data computes survival probabilities limited to a factor of the maximum
    observed data to avoid extending the survival curves too far beyond the range of observed data.

    Parameters
    ----------
    dt : DataTree
        Input data containing the predictive samples and observed data.
        Should contain groups specified by `group` and "observed_data",
        optionally including a censoring status variable in "constant_data".
        This censoring variable should be binary where 1 indicates an event
        occurred and 0 indicates censoring.
    var_names : str or list of str, optional
        One or more variables to be plotted.
    filter_vars : {None, "like", "regex"}, optional, default=None
        If None (default), interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior_predictive"
        Group to be plotted. Can be "posterior_predictive" or "prior_predictive".
    coords : dict, optional
        Coordinates to subset the data.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    num_samples : int, optional
        Number of samples to plot. Defaults to 100.
    extrapolation_factor : float, default 1.2
        Factor by which to limit the survival curves beyond the maximum observed time.
        Set to None to show the unaffected posterior predictive draws.
    plot_collection : PlotCollection, optional
        Existing plot collection to add to.
    backend : {"matplotlib", "bokeh", "plotly"}, optional
        Plotting backend to use.
    labeller : labeller, optional
        Labeller for plot titles and axes.
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * observed_km -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * predictive -> passed to :func:`~arviz_plots.visuals.ecdf_line`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

    **pc_kwargs
        Additional arguments passed to PlotCollection.

    Returns
    -------
    PlotCollection
        The plot collection containing the survival curve plot.

    Examples
    --------
    Plot Kaplan-Meier curves for posterior predictive checking:

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_censored, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('censored_cats')
        >>> plot_ppc_censored(dt)

    .. minigallery:: plot_ppc_censored

    References
    ----------
    .. [1] Kaplan, E. L., & Meier, P. Nonparametric estimation from incomplete observations.
           JASA, 53(282). (1958) https://doi.org/10.1080/01621459.1958.10501452
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

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if labeller is None:
        labeller = BaseLabeller()

    # Get predictive data
    predictive_dist = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    # Get observed data
    if "observed_data" in dt:
        observed_dist = process_group_variables_coords(
            dt,
            group="observed_data",
            var_names=var_names,
            filter_vars=filter_vars,
            coords=coords,
        )
        km_ds = kaplan_meier(dt, var_names=observed_dist.data_vars)
    else:
        observed_dist = None

    predictive_ds = generate_survival_curves(
        dt,
        var_names=predictive_dist.data_vars,
        group=group,
        num_samples=num_samples,
        extrapolation_factor=extrapolation_factor,
    )

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("overlay_ppc", ["sample"])
        pc_kwargs.setdefault("cols", "__variable__")
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, predictive_ds)

        plot_collection = PlotCollection.wrap(
            predictive_ds,
            backend=backend,
            **pc_kwargs,
        )

    aes_by_visuals.setdefault("predictive", ["overlay_ppc"])
    aes_by_visuals.setdefault("observed_km", plot_collection.aes_set)

    # Plot predictive survival curves
    predictive_kwargs = get_visual_kwargs(visuals, "predictive")
    if predictive_kwargs is not False:
        _, predictive_aes, predictive_ignore = filter_aes(
            plot_collection, aes_by_visuals, "predictive", sample_dims
        )
        if "color" not in predictive_aes:
            predictive_kwargs.setdefault("color", "C0")
        predictive_kwargs.setdefault("alpha", 0.7)

        plot_collection.map(
            ecdf_line,
            "predictive",
            data=predictive_ds,
            ignore_aes=predictive_ignore,
            **predictive_kwargs,
        )

    # Plot Kaplan-Meier curve
    observed_km_kwargs = get_visual_kwargs(
        visuals,
        "observed_km",
        False if group == "prior_predictive" or observed_dist is None else None,
    )
    if observed_km_kwargs is not False:
        _, observed_aes, observed_ignore = filter_aes(
            plot_collection, aes_by_visuals, "observed_km", sample_dims
        )
        if "color" not in observed_aes:
            observed_km_kwargs.setdefault("color", "B1")
        observed_km_kwargs.setdefault("linestyle", "C1")

        plot_collection.map(
            ecdf_line,
            "observed_km",
            data=km_ds,
            ignore_aes=observed_ignore,
            **observed_km_kwargs,
        )

    # Add labels
    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        xlabel_kwargs.setdefault("color", "B1")
        plot_collection.map(
            labelled_x,
            "xlabel",
            data=km_ds,
            subset_info=True,
            labeller=labeller,
            ignore_aes=plot_collection.aes_set,
            **xlabel_kwargs,
        )

    ylabel_kwargs = get_visual_kwargs(visuals, "ylabel")
    if ylabel_kwargs is not False:
        ylabel_kwargs.setdefault("text", "Survival Probability")
        ylabel_kwargs.setdefault("color", "B1")
        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=plot_collection.aes_set,
            **ylabel_kwargs,
        )

    # Add title
    title_kwargs = get_visual_kwargs(visuals, "title", False)
    if title_kwargs is not False:
        title_kwargs.setdefault("color", "B1")
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=plot_collection.aes_set,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    return plot_collection
