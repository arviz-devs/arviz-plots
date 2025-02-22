"""Contain functions for Bayesian Factor plotting."""

from importlib import import_module

from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_stats.bayes_factor import bayes_factor

from arviz_plots.plots.distplot import plot_dist


def plot_bf(
    dt,
    var_name,
    ref_val=0,
    kind=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Compute and plot the Bayesian Factor."""
    if kind is None:
        kind = rcParams["plot.density_kind"]
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

    # Compute Bayes Factor using the bayes_factor function
    bf, ref_vals = bayes_factor(dt, var_name, ref_val, return_ref_vals=True)

    # pylint: disable=all
    bf_10 = bf["BF10"]
    bf_01 = bf["BF01"]
    prior_at_ref_val = ref_vals["prior"]
    posterior_at_ref_val = ref_vals["posterior"]

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    try:
        plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    except ImportError as e:
        raise ImportError(f"Failed to import plotting backend '{backend}': {e}") from e

    color_cycle = pc_kwargs.get("color", plot_bknd.get_default_aes("color", 2, {}))
    if len(color_cycle) < 2:
        raise ValueError(
            f"Not enough values provided for color cycle, got {color_cycle} least 2 are needed."
        )

    if not isinstance(plot_kwargs, dict):
        plot_kwargs = {}

    if labeller is None:
        labeller = BaseLabeller()

    try:
        for group in ["prior", "posterior"]:
            plot_collection = plot_dist(
                dt,
                var_names=[var_name],
                group=group,
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
    except RuntimeError as e:
        raise RuntimeError(f"Error while plotting: {e}") from e

    return plot_collection
