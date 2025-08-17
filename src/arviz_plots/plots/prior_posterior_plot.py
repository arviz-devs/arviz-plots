"""Contain functions for Bayes Factor plotting."""

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import extract, rcParams
from xarray import concat

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    get_visual_kwargs,
    process_group_variables_coords,
    set_wrap_layout,
)


def plot_prior_posterior(
    dt,
    var_names=None,
    filter_vars=None,
    group=None,  # pylint: disable=unused-argument
    coords=None,
    sample_dims=None,
    kind=None,
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
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[
        Literal["dist", "credible_interval", "point_estimate"], Mapping[str, Any] | xr.Dataset
    ] = None,
    **pc_kwargs,
):
    r"""Plot 1D marginal densities for prior and posterior.

    The Bayes factor is estimated by comparing a model (H1) against a model
    in which the parameter of interest has been restricted to be a point-null (H0)
    This computation assumes the models are nested and thus H0 is a special case of H1.

    Parameters
    ----------
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default=None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : None
        This argument is ignored. Have it here for compatibility with other plotting functions.
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. The prior and posterior groups are combined creating a new
        dimension "group". By default, there is an aesthetic mapping from group to color.
        Valid keys are the same as for `visuals`.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * dist -> depending on the value of `kind` passed to:

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * legend -> passed to :class:`arviz_plots.PlotCollection.add_legend`

    stats : mapping, optional
        Valid keys are:

        * dist -> passed to kde, ecdf, ...

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    Examples
    --------
    Select two variables and plot them with a ecdf.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_prior_posterior, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_prior_posterior(dt, var_names=["mu", "tau"], kind="ecdf")


    .. minigallery:: plot_prior_posterior
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
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if not isinstance(visuals, dict):
        visuals = {}

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    prior_size = np.prod([dt.prior.sizes[dim] for dim in sample_dims])
    posterior_size = np.prod([dt.posterior.sizes[dim] for dim in sample_dims])
    num_samples = min(prior_size, posterior_size)

    ds_prior = (
        extract(dt, group="prior", num_samples=num_samples, random_seed=0, keep_dataset=True)
        .drop_vars(sample_dims + ["sample"])
        .assign_coords(sample=("sample", np.arange(num_samples)))
    )
    ds_posterior = (
        extract(dt, group="posterior", num_samples=num_samples, random_seed=0, keep_dataset=True)
        .drop_vars(sample_dims + ["sample"])
        .assign_coords(sample=("sample", np.arange(num_samples)))
    )

    distribution = concat([ds_prior, ds_posterior], dim="group").assign_coords(
        {"group": ["prior", "posterior"]}
    )

    distribution = process_group_variables_coords(
        distribution,
        group=None,
        var_names=var_names,
        filter_vars=filter_vars,
        coords=coords,
    )

    if len(sample_dims) > 1:
        # sample dims will have been stacked and renamed by `extract`
        sample_dims = ["sample"]

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("color", ["group"])
        pc_kwargs.setdefault("col_wrap", 4)
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]
            + [dim for dim in distribution.dims if dim not in sample_dims + ["group"]],
        )

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, distribution)

        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    visuals.setdefault("credible_interval", False)
    visuals.setdefault("point_estimate", False)
    visuals.setdefault("point_estimate_text", False)

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    if kind == "hist":
        visuals.setdefault("dist", {})
        visuals.setdefault("remove_axis", True)

    plot_collection = plot_dist(
        distribution,
        var_names=None,
        group=None,
        coords=None,
        sample_dims=sample_dims,
        kind=kind,
        point_estimate=None,
        ci_kind=None,
        ci_prob=None,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats,
        **pc_kwargs,
    )

    legend_kwargs = get_visual_kwargs(visuals, "legend")
    if legend_kwargs is not False:
        legend_kwargs.setdefault("dim", ["group"])
        plot_collection.add_legend(**legend_kwargs)

    return plot_collection
