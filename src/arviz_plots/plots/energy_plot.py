"""Energy plot code."""
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz_base import convert_to_dataset, rcParams

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import filter_aes, get_visual_kwargs, set_grid_layout
from arviz_plots.visuals import labelled_title, labelled_y, scatter_xy, vline


def plot_energy(
    dt,
    *,
    sample_dims=None,
    kind=None,
    show_bfmi=True,
    threshold=0.3,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "title",
            "bfmi_points",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "dist",
            "title",
            "legend",
            "remove_axis",
            "bfmi_points",
            "ref_line",
            "title",
            "ylabel",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    r"""Plot energy distributions and bfmi from gradient-based algorithms.

    Generate a figure with the marginal energy distribution and the energy transition
    distribution. Optionally, include a BFMI panel to inspect chain-wise Bayesian Fraction
    of Missing Information values. Values below the threshold indicate poor exploration
    of the energy distribution.

    For details on BFMI and energy diagnostics see [1]_ for a more practical overview check
    the EABM chapter on MCMC diagnostic `of gradient-based algorithms <https://arviz-devs.github.io/EABM/Chapters/MCMC_diagnostics.html#diagnosis-of-gradient-based-algorithms>`_.


    Parameters
    ----------
    dt : DataTree
        ``sample_stats`` group with an ``energy`` variable is mandatory.
    sample_dims : sequence of str, optional
        Dimensions to consider as sample dimensions when computing BFMI.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    show_bfmi : bool, default True
        Whether to include the BFMI scatter plot. If ``False``, only the energy plot will be shown.
    threshold : float, default 0.3
        Reference threshold for BFMI values, values below this indicate poor exploration of the
        energy distribution.
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
          * "hist" -> passed to :func:`~arviz_plots.visuals.step_hist`
          * "dot" -> passed to :func:`~arviz_plots.visuals.scatter_xy`

        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * legend -> passed to :class:`arviz_plots.PlotCollection.add_legend`
        * remove_axis -> not passed anywhere, can only be ``False`` to skip calling this function
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * bfmi_points -> passed to :func:`~arviz_plots.visuals.scatter_xy` for BFMI scatter plot
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y` for BFMI column y-axis label
        * face -> :term:`visual` that fills the area under the energy distributions.

          Defaults to True. Depending on the value of `kind` it is passed to:

          * "kde" or "ecdf" -> passed to :func:`~arviz_plots.visuals.fill_between_y`
          * "hist" -> passed to :func:`~arviz_plots.visuals.hist`
          * dot -> ignored

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
    Plot an energy plot using ecdf for the energy distributions.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_energy, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> data = load_arviz_data('non_centered_eight')
        >>> plot_energy(data, kind="ecdf")


    .. minigallery:: plot_energy

    References
    ----------
    .. [1] Betancourt. Diagnosing Suboptimal Cotangent Disintegrations in
        Hamiltonian Monte Carlo. (2016) https://arxiv.org/abs/1604.00695
    """  # pylint: disable=line-too-long
    if kind is None:
        kind = rcParams["plot.density_kind"]
    if visuals is None:
        visuals = {}
    else:
        visuals = visuals.copy()
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    if kind not in ("kde", "hist", "ecdf", "dot"):
        raise ValueError("kind must be either 'kde', 'hist', 'ecdf' or 'dot'")

    energy_ds, bfmi_ds = _get_energy_ds(dt, sample_dims=sample_dims)

    if backend is None:
        backend = rcParams["plot.backend"]
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        if show_bfmi:
            new_ds = energy_ds.expand_dims(column=2).assign_coords(column=["bfmi", "energy"])
            pc_kwargs["figure_kwargs"].setdefault("width_ratios", [1, 3])
            num_cols = 2
        else:
            new_ds = energy_ds.expand_dims(column=1).assign_coords(column=["energy"])
            num_cols = 1

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs["aes"].setdefault("color", ["energy"])
        pc_kwargs = set_grid_layout(pc_kwargs, plot_bknd, new_ds, num_cols=num_cols, num_rows=1)

        plot_collection = PlotCollection.grid(
            new_ds,
            backend=backend,
            **pc_kwargs,
        )

    visuals.setdefault("credible_interval", False)
    visuals.setdefault("point_estimate", False)
    visuals.setdefault("point_estimate_text", False)
    visuals.setdefault("face", True)

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    # Energy distributions plot
    plot_collection.coords = {"column": "energy"}
    plot_collection = plot_dist(
        energy_ds,
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
    )
    plot_collection.coords = None

    # legend for energy distributions
    legend_kwargs = get_visual_kwargs(visuals, "legend")
    if legend_kwargs is not False:
        legend_kwargs.setdefault("dim", ["energy"])
        legend_kwargs.setdefault("title", "")
        plot_collection.add_legend(**legend_kwargs)

    if show_bfmi:
        # Scatter plot of BFMI values
        bfmi_ms_kwargs = get_visual_kwargs(visuals, "bfmi_points")
        if bfmi_ms_kwargs is not False:
            _, _, bfmi_ignore = filter_aes(plot_collection, aes_by_visuals, "bfmi_points", [])

            bfmi_ms_kwargs.setdefault("color", "B1")

            plot_collection.coords = {"column": "bfmi"}
            plot_collection.map(
                scatter_xy,
                "bfmi_points",
                data=bfmi_ds,
                ignore_aes=bfmi_ignore,
                **bfmi_ms_kwargs,
            )
            plot_collection.coords = None

        # Reference line for BFMI threshold
        ref_line_kwargs = get_visual_kwargs(visuals, "ref_line")
        if ref_line_kwargs is not False:
            _, ref_aes, ref_ignore = filter_aes(
                plot_collection, aes_by_visuals, "ref_line", sample_dims
            )
            if "color" not in ref_aes:
                ref_line_kwargs.setdefault("color", "B2")
            if "linestyle" not in ref_aes:
                ref_line_kwargs.setdefault("linestyle", "C1")

            # Wrap threshold into an xr.Dataset for PlotCollection.map
            ref_ds = xr.Dataset({"ref_line": xr.DataArray(threshold)})
            plot_collection.coords = {"column": "bfmi"}
            plot_collection.map(
                vline, "ref_line", data=ref_ds, ignore_aes=ref_ignore, **ref_line_kwargs
            )
            plot_collection.coords = None

        # Add title for BFMI plot
        title_kwargs = get_visual_kwargs(visuals, "title")
        if title_kwargs is not False:
            _, title_aes, title_ignore = filter_aes(
                plot_collection, aes_by_visuals, "title", sample_dims
            )
            if "color" not in title_aes:
                title_kwargs.setdefault("color", "B1")
            plot_collection.coords = {"column": "bfmi"}
            plot_collection.map(
                labelled_title,
                "title",
                text="BFMI",
                ignore_aes=title_ignore,
                subset_info=True,
                labeller=labeller,
                **title_kwargs,
            )
            plot_collection.coords = None

        # Add ylabel for BFMI plot
        ylabel_kwargs = get_visual_kwargs(visuals, "ylabel")
        if ylabel_kwargs is not False:
            ylabel_kwargs.setdefault("text", "Chain")
            _, _, ylabel_ignore = filter_aes(plot_collection, {}, "ylabel", [])
            plot_collection.coords = {"column": "bfmi"}
            plot_collection.map(
                labelled_y,
                "ylabel",
                ignore_aes=ylabel_ignore,
                **ylabel_kwargs,
            )
            plot_collection.coords = None

    return plot_collection


def _get_energy_ds(dt, sample_dims):
    """Extract energy and BFMI data from DataTree.

    Returns
    -------
    energy_ds : Dataset
        Dataset with Energy variable containing marginal and transition energy
    bfmi_ds : Dataset
        Dataset with bfmi variable containing BFMI values and chain indices
    """
    energy = dt["sample_stats"].energy.values
    bfmi_vals = dt.sample_stats["energy"].azstats.bfmi(sample_dims=sample_dims)
    n_chains = len(bfmi_vals)
    chain_indices = np.arange(n_chains)

    bfmi_ds = xr.Dataset(
        {
            "bfmi": xr.DataArray(
                np.column_stack([bfmi_vals.values, chain_indices]),
                dims=["chain", "plot_axis"],
                coords={"chain": bfmi_vals.chain, "plot_axis": ["x", "y"]},
            )
        }
    )

    energy_ds = convert_to_dataset(
        {"Energy": np.dstack([energy - energy.mean(), np.diff(energy, append=np.nan)])},
        coords={"Energy_dim_0": ["marginal", "transition"]},
    ).rename({"Energy_dim_0": "energy"})

    return energy_ds, bfmi_ds
