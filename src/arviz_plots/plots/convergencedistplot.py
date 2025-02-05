"""Convergence diagnostic distribution plot code."""
import warnings
from copy import copy
from importlib import import_module

import arviz_stats  # pylint: disable=unused-import
import xarray as xr
from arviz_base import rcParams

from arviz_plots.plots.distplot import plot_dist
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import vline


def plot_convergence_dist(
    dt,
    diagnostics=None,
    ref_line=True,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind="ecdf",
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot the distribution of convergence diagnostics (ESS and/or R-hat).

    Parameters
    ----------
    dt : DataTree
        Input data
    diagnostics : list of str, optional
        List of diagnostics to plot. Defaults to ["ess_bulk", "ess_tail", "rhat_rank"].
        Valid diagnostics are "rhat_rank", "rhat_folded", "rhat_z_scale", "rhat_split",
        "rhat_identity", "ess_bulk", "ess_tail", "ess_mean", "ess_sd", "ess_quantile",
        "ess_local", "ess_median", "ess_mad", "ess_z_scale", "ess_folded" and "ess_identity".
    ref_line : bool, default True
        Whether to plot a reference line for the recommended value of each diagnostic.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default=None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
        The group for which to compute the convergence diagnostics.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the distribution of diagnostics. Default to ecdf
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : sequence of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.
        By default, no mappings are defined for this plot.
    plot_kwargs : mapping of {str : mapping or False}, optional
        Valid keys are:

        * One of "kde", "ecdf", "dot" or "hist", matching the `kind` argument.

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * ref_line -> passed to :func:`~arviz_plots.visuals.vline`
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
    Select a single variable and specify diagnostics

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_convergence_dist, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> radon = load_arviz_data('radon')
        >>> plot_convergence_dist(
        >>>     radon,
        >>>     var_names=["za_county"],
        >>>     diagnostics=["rhat", "ess_tail"]
        >>> )

    Some ess methods accepts a probability argument

    .. plot::
        :context: close-figs

        >>> plot_convergence_dist(
        >>>     radon,
        >>>     var_names=["za_county"],
        >>>     diagnostics=[
        >>>         "ess_tail(0.1, 0.9)",
        >>>         "ess_local(0.1, 0.9)",
        >>>         "ess_quantile(0.9)"
        >>>     ]
        >>> )

    .. minigallery:: plot_convergence_dist

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    sample_dims = list(sample_dims)
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()

    ref_line_kwargs = copy(plot_kwargs.get("ref_line", {}))
    if ref_line_kwargs is False:
        raise ValueError(
            "plot_kwargs['ref_line'] can't be False, use ref_line=False to remove this element"
        )

    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    if diagnostics is None:
        diagnostics = ["ess_bulk", "ess_tail", "rhat"]
    elif isinstance(diagnostics, str):
        diagnostics = [diagnostics]

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend
    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    dt = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    distribution = _compute_diagnostics(dt, diagnostics, sample_dims)

    plot_kwargs.setdefault("credible_interval", False)
    plot_kwargs.setdefault("point_estimate", False)
    plot_kwargs.setdefault("point_estimate_text", False)

    plot_collection = plot_dist(
        distribution,
        var_names=None,
        filter_vars=None,
        group=None,
        coords=None,
        # _compute_diagnostics returns output with only this dimension
        # it is the one we want reduced in plot_dist to show the distributions
        sample_dims=["label"],
        kind=kind,
        point_estimate=point_estimate,
        ci_kind=ci_kind,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_map=aes_map,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
        pc_kwargs=pc_kwargs,
    )

    if ref_line:
        _, ref_aes, ref_ignore = filter_aes(plot_collection, aes_map, "ref_line", sample_dims)
        if "color" not in ref_aes:
            ref_line_kwargs.setdefault("color", "black")
        if "linestyle" not in ref_aes:
            default_linestyle = plot_bknd.get_default_aes("linestyle", 2, {})[1]
            ref_line_kwargs.setdefault("linestyle", default_linestyle)
        if "alpha" not in ref_aes:
            ref_line_kwargs.setdefault("alpha", 0.5)

        ess_ref = dt.sizes["chain"] * 100
        # is this valid for all r_hat methods? Do we want to correct for multiple comparisons?
        r_hat_ref = 1.01
        ref_ds = xr.Dataset(
            {
                diagnostic: ess_ref if "ess" in diagnostic else r_hat_ref
                for diagnostic in distribution.data_vars
            }
        )
        plot_collection.map(
            vline, "ref_line", data=ref_ds, ignore_aes=ref_ignore, **ref_line_kwargs
        )

    return plot_collection


def _compute_diagnostics(dt, diagnostics, sample_dims):
    diagnostic_values = {}
    for diagnostic in diagnostics:
        if "ess" in diagnostic:
            prob = None
            method = diagnostic.split("_", 1)[1].split("(", 1)[0]
            if method in {"tail", "quantile", "local"} and "(" in diagnostic:
                prob = [float(p) for p in diagnostic.split("(", 1)[1].rstrip(")").split(", ")]
            diagnostic_values[diagnostic] = dt.azstats.ess(
                method=method, prob=prob, dims=sample_dims
            ).to_stacked_array("label", sample_dims=[])
        elif "rhat" in diagnostic:
            kwargs = {"dims": sample_dims}
            if diagnostic != "rhat":
                method = diagnostic.split("_", 1)[1]
                kwargs.update({"method": method})
            diagnostic_values[diagnostic] = dt.azstats.rhat(**kwargs).to_stacked_array(
                "label", sample_dims=[]
            )
        else:
            warnings.warn(f"{diagnostic} is not recognized as a valid diagnostic")
    return xr.Dataset(diagnostic_values)
