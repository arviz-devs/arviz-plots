"""Energy plot code."""
import numpy as np
from arviz_base import convert_to_dataset, rcParams

from arviz_plots.plots.distplot import plot_dist


def plot_energy(
    dt,
    bfmi=False,
    kind=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    r"""Plot transition distribution and marginal energy distribution in HMC algorithms.

    This may help to diagnose poor exploration by gradient-based algorithms like HMC or NUTS.
    The energy function in HMC can identify posteriors with heavy tailed distributions, that
    in practice are challenging for sampling.

    This plot is in the style of the one used in [1]_.

    Parameters
    ----------
    dt : DataTree
        ``sample_stats`` group with an ``energy`` variable is mandatory.
    bfmi : bool
        Whether to the plot the value of the estimated Bayesian fraction of missing
        information. Defaults to False. Not implemented yet.
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
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
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

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
    Plot a default energy plot

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_energy, style
        >>> style.use("arviz-clean")
        >>> from arviz_base import load_arviz_data
        >>> schools = load_arviz_data('centered_eight')
        >>> plot_energy(schools)


    .. minigallery:: plot_energy

    """
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

    new_ds = _get_energy_ds(dt)

    sample_dims = ["chain", "draw"]
    if not all(dim in new_ds.dims for dim in sample_dims):
        raise ValueError("Both 'chain' and 'draw' dimensions must be present in the dataset")

    pc_kwargs.setdefault("cols", None)
    pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
    pc_kwargs["aes"].setdefault("color", ["energy"])
    plot_kwargs.setdefault("credible_interval", False)
    plot_kwargs.setdefault("point_estimate", False)
    plot_kwargs.setdefault("point_estimate_text", False)
    plot_kwargs.setdefault("title", False)

    plot_collection = plot_dist(
        new_ds,
        var_names=None,
        filter_vars=None,
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
        aes_map=aes_map,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
        pc_kwargs=pc_kwargs,
    )

    if backend == "matplotlib":  ## remove this when we have a better way to handle legends
        plot_collection.add_legend("energy", loc="outside right upper")

    if bfmi:
        raise NotImplementedError("BFMI is not implemented yet")

    return plot_collection


def _get_energy_ds(dt):
    energy = dt["sample_stats"].energy.values
    return convert_to_dataset(
        {"energy_": np.dstack([energy - energy.mean(), np.diff(energy, append=np.nan)])},
        coords={"energy__dim_0": ["marginal", "transition"]},
    ).rename({"energy__dim_0": "energy"})
