"""Energy plot code."""
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset, rcParams

from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import get_visual_kwargs


def plot_energy(
    dt,
    bfmi=False,
    kind=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "dist",
            "title",
            "legend",
            "remove_axis",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
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
        Whether to add to the legend the estimated Bayesian fraction of missing information.
        Defaults to False.
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * dist -> depending on the value of `kind` passed to:

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.step_hist`
          * "dot" -> passed to :func:`~arviz_plots.visuals.scatter_xy`

        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * legend -> passed to :class:`arviz_plots.PlotCollection.add_legend`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function

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
    Plot a default energy plot

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_energy, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> schools = load_arviz_data('centered_eight')
        >>> plot_energy(schools)


    .. minigallery:: plot_energy

    References
    ----------
    .. [1] Betancourt. Diagnosing Suboptimal Cotangent Disintegrations in
        Hamiltonian Monte Carlo. (2016) https://arxiv.org/abs/1604.00695
    """
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()

    if kind not in ("kde", "hist", "ecdf", "dot"):
        raise ValueError("kind must be either 'kde', 'hist', 'ecdf' or 'dot'")

    new_ds = _get_energy_ds(dt)

    sample_dims = ["chain", "draw"]
    if not all(dim in new_ds.dims for dim in sample_dims):
        raise ValueError("Both 'chain' and 'draw' dimensions must be present in the dataset")

    pc_kwargs.setdefault("cols", None)
    pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
    pc_kwargs["aes"].setdefault("color", ["energy"])
    visuals.setdefault("credible_interval", False)
    visuals.setdefault("point_estimate", False)
    visuals.setdefault("point_estimate_text", False)
    visuals.setdefault("title", False)

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
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats,
        **pc_kwargs,
    )

    # legend
    legend_kwargs = get_visual_kwargs(visuals, "legend")
    if legend_kwargs is not False:
        legend_kwargs.setdefault("dim", ["energy"])

        line_break = "<br>" if backend == "plotly" else "\n"

        if bfmi:
            bfmi_values = dt.sample_stats["energy"].azstats.bfmi()
            bfmi_text = f"BFMI{line_break}" + line_break.join(
                f"        chain {chain}  {_format_bfmi(bfmi_values[chain])}"
                for chain in range(len(bfmi_values))
            )
            legend_kwargs.setdefault("title", bfmi_text + f"{line_break}{line_break}Energy")

        plot_collection.add_legend(**legend_kwargs)

    return plot_collection


def _get_energy_ds(dt):
    energy = dt["sample_stats"].energy.values
    return convert_to_dataset(
        {"energy_": np.dstack([energy - energy.mean(), np.diff(energy, append=np.nan)])},
        coords={"energy__dim_0": ["marginal", "transition"]},
    ).rename({"energy__dim_0": "energy"})


def _format_bfmi(value):
    formatted = f"{value:.2f}"
    if value < 0.3:
        return f"{formatted} ⚠"
    return formatted
