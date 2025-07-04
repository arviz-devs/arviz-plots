"""Posterior/prior predictive check using densities."""

import warnings
from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords, set_wrap_layout
from arviz_plots.visuals import ecdf_line, hist, line_xy


def plot_ppc_dist(
    dt,
    data_pairs=None,
    var_names=None,
    filter_vars=None,
    group="posterior_predictive",
    coords=None,
    sample_dims=None,
    kind=None,
    num_samples=50,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal["predictive_dist", "observed_dist", "title"], Sequence[str]
    ] = None,
    visuals: Mapping[
        Literal["predictive_dist", "observed_dist", "title", "remove_axis"],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[
        Literal["predictive_dist", "observed_dist"], Mapping[str, Any] | xr.Dataset
    ] = None,
    **pc_kwargs,
):
    """
    Plot 1D marginals for the posterior/prior predictive distribution and the observed data.

    Parameters
    ----------
    dt : DataTree
        If group is "posterior_predictive", it should contain the ``posterior_predictive`` and
        ``observed_data`` groups. If group is "prior_predictive", it should contain the
        ``prior_predictive`` group.
    data_pairs : dict, optional
        Dictionary of keys prior/posterior predictive data and values observed data variable names.
        If None, it will assume that the observed data and the predictive data have
        the same variable name.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default=None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str,
        Group to be plotted. Defaults to "posterior_predictive".
        It could also be "prior_predictive".
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    num_samples : int, optional
        Number of samples to plot. Defaults to 100.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals` except "remove_axis".

        With a single model, no aesthetic mappings are generated by default,
        each variable+coord combination gets a :term:`plot` but they all look the same,
        unless there are user provided aesthetic mappings.
        With multiple models, ``plot_dist`` maps "color" and "y" to the "model" dimension.

        By default, all aesthetics but "y" are mapped to the distribution representation,
        and if multiple models are present, "color" and "y" are mapped to the
        credible interval and the point estimate.

        When "point_estimate" key is provided but "point_estimate_text" isn't,
        the values assigned to the first are also used for the second.
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * predictive_dist, observed_dist -> passed to a function that depends on
          the `kind` argument.

          - `kind="kde"` -> passed to :func:`~arviz_plots.visuals.line_xy`
          - `kind="ecdf"` -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          - `kind="hist"` -> passed to :func: `~arviz_plots.visuals.hist`

        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

        observed_dist defaults to False, no observed data is plotted, if group is
        "prior_predictive". Pass an (empty) mapping to plot the observed data.

    stats : mapping, optional
        Valid keys are:

        * predictive_dist, observed_dist -> passed to kde, ecdf, ...

    **pc_kwargs
        Passed to :meth:`~arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    See Also
    --------
    :ref:`plots_intro` :
        General introduction to batteries-included plotting functions, common use and logic overview

    Examples
    --------
    Make a plot of the posterior predictive distribution vs the observed data.
    We used an ECDF representation customized the colors.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ppc_dist, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> radon = load_arviz_data('radon')
        >>> pc = plot_ppc_dist(
        >>>     radon,
        >>>     kind="ecdf",
        >>>     visuals={
        >>>         "predictive_dist": {"color":"C1"},
        >>>         "observed_dist": {"color":"C3"}
        >>>     },
        >>> )

    .. minigallery:: plot_ppc_dist
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

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    rng = np.random.default_rng(4214)

    pp_dims = list(sample_dims) + [dims for dims in dt[group].dims if dims not in sample_dims]

    if data_pairs is None:
        data_pairs = (var_names, var_names)
    else:
        data_pairs = (list(data_pairs.keys()), list(data_pairs.values()))

    predictive_dist = process_group_variables_coords(
        dt, group=group, var_names=data_pairs[0], filter_vars=filter_vars, coords=coords
    )
    predictive_types = [
        predictive_dist[var].values.dtype.kind == "i" for var in predictive_dist.data_vars
    ]

    if "observed_data" in dt:
        observed_dist = process_group_variables_coords(
            dt,
            group="observed_data",
            var_names=data_pairs[1],
            filter_vars=filter_vars,
            coords=coords,
        )
        observed_types = [
            observed_dist[var].values.dtype.kind == "i" for var in observed_dist.data_vars
        ]
    else:
        observed_types = []

    if any(predictive_types + observed_types):
        warnings.warn(
            "Detected at least one discrete variable.\n"
            "Consider using plot_ppc variants specific for discrete data, "
            "such as plot_ppc_pava or plot_ppc_rootogram.",
            UserWarning,
            stacklevel=2,
        )

    # Select a random subset of samples
    n_pp_samples = np.prod(
        [predictive_dist.sizes[dim] for dim in sample_dims if dim in predictive_dist.dims]
    )
    if num_samples > n_pp_samples:
        num_samples = n_pp_samples
        warnings.warn("num_samples is larger than the number of predictive samples.")

    pp_sample_ix = rng.choice(n_pp_samples, size=num_samples, replace=False)
    predictive_dist = predictive_dist.stack(sample=sample_dims).isel(sample=pp_sample_ix)

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("overlay_ppc", ["sample"])
        pc_kwargs.setdefault("cols", "__variable__")

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, predictive_dist)

        plot_collection = PlotCollection.wrap(
            predictive_dist,
            backend=backend,
            **pc_kwargs,
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    if labeller is None:
        labeller = BaseLabeller()

    # We don't want credible_interval or point_estimate to be mapped to the density representation
    visuals.setdefault("credible_interval", False)
    visuals.setdefault("point_estimate", False)
    visuals.setdefault("point_estimate_text", False)
    visuals.setdefault("rug_plot", False)

    # Plot the predictive density
    pred_density_kwargs = copy(visuals.get("predictive_dist", {}))
    if pred_density_kwargs is not False:
        visuals.setdefault("dist", pred_density_kwargs)
        visuals["dist"].setdefault("alpha", 0.3)
        if kind == "hist":
            if visuals["dist"] is not False:
                visuals["dist"].setdefault("edgecolor", None)

        plot_collection = plot_dist(
            predictive_dist,
            group=group,
            sample_dims=pp_dims,
            kind=kind,
            visuals=visuals,
            aes_by_visuals=aes_by_visuals,
            pc_kwargs=pc_kwargs,
            plot_collection=plot_collection,
            stats={"dist": stats.get("predictive_dist", {})},
        )
        plot_collection.rename_visuals(dist="predictive_dist")

    # Plot the observed density
    observed_density_kwargs = copy(
        visuals.get("observed_dist", False if group == "prior_predictive" else {})
    )

    if observed_density_kwargs is not False:
        observed_stats_kwargs = stats.get("observed_dist", {}).copy()
        observed_density_kwargs.setdefault("color", "black")
        if kind == "hist":
            observed_density_kwargs.setdefault("alpha", 0.3)
            observed_density_kwargs.setdefault("edgecolor", None)

        _, _, observed_ignore = filter_aes(
            plot_collection, aes_by_visuals, "observed_dist", sample_dims
        )

        if kind == "kde":
            dt_observed = observed_dist.azstats.kde(dim=pp_dims, **observed_stats_kwargs)

            plot_collection.map(
                line_xy,
                "observed_dist",
                data=dt_observed,
                ignore_aes=observed_ignore,
                **observed_density_kwargs,
            )

        if kind == "hist":
            observed_stats_kwargs.setdefault("density", True)
            dt_observed = observed_dist.azstats.histogram(dim=pp_dims, **observed_stats_kwargs)

            plot_collection.map(
                hist,
                "observed_dist",
                data=dt_observed,
                ignore_aes=observed_ignore,
                **observed_density_kwargs,
            )

        if kind == "ecdf":
            dt_observed = observed_dist.azstats.ecdf(**observed_stats_kwargs)
            plot_collection.map(
                ecdf_line,
                "observed_dist",
                data=dt_observed,
                ignore_aes=observed_ignore,
                **observed_density_kwargs,
            )

    return plot_collection
