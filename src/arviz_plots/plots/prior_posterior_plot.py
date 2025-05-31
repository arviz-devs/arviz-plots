"""Contain functions for Bayes Factor plotting."""

from importlib import import_module

import numpy as np
from arviz_base import extract, rcParams
from xarray import concat

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import process_group_variables_coords, set_wrap_layout


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
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
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
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.
    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * One of "kde", "ecdf", "dot" or "hist", matching the `kind` argument.

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

    stats_kwargs : mapping, optional
        Valid keys are:

        * density -> passed to kde, ecdf, ...

    pc_kwargs : mapping
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
    if stats_kwargs is None:
        stats_kwargs = {}
    else:
        stats_kwargs = stats_kwargs.copy()
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if not isinstance(plot_kwargs, dict):
        plot_kwargs = {}

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

    distribution = concat([ds_prior, ds_posterior], dim="Group").assign_coords(
        {"Group": ["prior", "posterior"]}
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
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("color", ["Group"])
        pc_kwargs.setdefault("col_wrap", 4)
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]
            + [dim for dim in distribution.dims if dim not in sample_dims + ["Group"]],
        )

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, distribution)

        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    plot_kwargs.setdefault("credible_interval", False)
    plot_kwargs.setdefault("point_estimate", False)
    plot_kwargs.setdefault("point_estimate_text", False)

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    if kind == "hist":
        plot_kwargs.setdefault("hist", {})
        plot_kwargs.setdefault("remove_axis", True)
        if plot_kwargs["hist"] is not False:
            plot_kwargs["hist"].setdefault("step", True)
            stats_kwargs.setdefault("density", {"density": True})

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
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
        pc_kwargs=pc_kwargs,
    )

    plot_collection.add_legend("Group")

    return plot_collection
