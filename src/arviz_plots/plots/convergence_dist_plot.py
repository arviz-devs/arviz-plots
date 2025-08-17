"""Convergence diagnostic distribution plot code."""
import warnings
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import arviz_stats  # pylint: disable=unused-import
import xarray as xr
from arviz_base import rcParams

from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.utils import (
    filter_aes,
    get_contrast_colors,
    get_visual_kwargs,
    process_group_variables_coords,
)
from arviz_plots.visuals import vline


def plot_convergence_dist(
    dt,
    diagnostics=None,
    grouped=True,
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
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "ref_line",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "dist",
            "ref_line",
            "title",
            "remove_axis",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    """Plot the distribution of convergence diagnostics (ESS and/or R-hat).

    By default all variables are grouped together and one plot per diagnostic is created.
    If you are interested in representing individual (multidimensional variables) pass them
    in `var_names`.

    Information on how the diagnostics are computed can be found in [1]_.

    Parameters
    ----------
    dt : DataTree
        Input data
    diagnostics : list of str, optional
        List of diagnostics to plot. Defaults to ["ess_bulk", "ess_tail", "rhat_rank"].
        Valid diagnostics are "rhat_rank", "rhat_folded", "rhat_z_scale", "rhat_split",
        "rhat_identity", "ess_bulk", "ess_tail", "ess_mean", "ess_sd", "ess_quantile",
        "ess_local", "ess_median", "ess_mad", "ess_z_scale", "ess_folded" and "ess_identity".
    grouped: bool, optional
        Whether to plot all variables listed in ``var_names`` together (True)
        or separately (False). Defaults to True.
        If False, all variables listed in ``var_names`` must be multidimensional.
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
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals` except for "remove_axis"
        By default, no mappings are defined for this plot.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * dist -> depending on the value of `kind` passed to:

          * "kde" -> passed to :func:`~arviz_plots.visuals.line_xy`
          * "ecdf" -> passed to :func:`~arviz_plots.visuals.ecdf_line`
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * ref_line -> passed to :func:`~arviz_plots.visuals.vline`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
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

    Select two variables and plot them separately

    .. plot::
        :context: close-figs

        >>> plot_convergence_dist(
        >>>     radon,
        >>>     var_names=["za_county", "a"],
        >>>     grouped=False,
        >>> )

    .. minigallery:: plot_convergence_dist


    References
    ----------
    .. [1] Vehtari et al. *Rank-normalization, folding, and localization: An improved Rhat for
        assessing convergence of MCMC*. Bayesian Analysis. 16(2) (2021)
        https://doi.org/10.1214/20-BA1221. arXiv preprint https://arxiv.org/abs/1903.08008
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

    ref_line_kwargs = get_visual_kwargs(visuals, "ref_line")
    if ref_line_kwargs is False:
        raise ValueError(
            "visuals['ref_line'] can't be False, use ref_line=False to remove this element"
        )

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

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
    bg_color = plot_bknd.get_background_color()
    contrast_color = get_contrast_colors(bg_color=bg_color)

    dt = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    distribution = _compute_diagnostics(dt, diagnostics, sample_dims, grouped)

    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()

        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if grouped is False:
            pc_kwargs.setdefault("col_wrap", len(diagnostics))

    if grouped:
        plot_dist_sample_dims = ["label"]
        plot_dist_var_names = None
    else:
        plot_dist_sample_dims = [dim for dim in dt.dims if dim not in sample_dims]
        plot_dist_var_names = var_names

    visuals.setdefault("credible_interval", False)
    visuals.setdefault("point_estimate", False)
    visuals.setdefault("point_estimate_text", False)

    plot_collection = plot_dist(
        distribution,
        var_names=plot_dist_var_names,
        filter_vars=None,
        group=None,
        coords=None,
        # _compute_diagnostics returns output with only this dimension
        # it is the one we want reduced in plot_dist to show the distributions
        sample_dims=plot_dist_sample_dims,
        kind=kind,
        point_estimate=point_estimate,
        ci_kind=ci_kind,
        ci_prob=ci_prob,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals=aes_by_visuals,
        visuals=visuals,
        stats=stats,
        **pc_kwargs,
    )

    if ref_line:
        _, ref_aes, ref_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ref_line", sample_dims
        )
        if "color" not in ref_aes:
            ref_line_kwargs.setdefault("color", contrast_color)
        if "linestyle" not in ref_aes:
            default_linestyle = plot_bknd.get_default_aes("linestyle", 2, {})[1]
            ref_line_kwargs.setdefault("linestyle", default_linestyle)
        if "alpha" not in ref_aes:
            ref_line_kwargs.setdefault("alpha", 0.5)

        ess_ref = dt.sizes["chain"] * 100
        # is this valid for all r_hat methods? Do we want to correct for multiple comparisons?
        r_hat_ref = 1.01
        if grouped:
            ref_ds = xr.Dataset(
                {
                    diagnostic: ess_ref if "ess" in diagnostic else r_hat_ref
                    for diagnostic in diagnostics
                }
            )
        else:
            ref_values = [
                ess_ref if "ess" in diagnostic else r_hat_ref for diagnostic in diagnostics
            ]
            ref_ds = xr.Dataset(
                {
                    var: xr.DataArray(ref_values, dims="diagnostic")
                    for var in distribution.data_vars
                },
                coords={"diagnostic": diagnostics},
            )

        plot_collection.map(
            vline, "ref_line", data=ref_ds, ignore_aes=ref_ignore, **ref_line_kwargs
        )

    return plot_collection


def _compute_diagnostics(dt, diagnostics, sample_dims, grouped):
    diagnostic_dict = {}
    for diagnostic in diagnostics:
        if "ess" in diagnostic:
            prob = None
            method = diagnostic.split("_", 1)[1].split("(", 1)[0]
            if method in {"tail", "quantile", "local"} and "(" in diagnostic:
                prob = [float(p) for p in diagnostic.split("(", 1)[1].rstrip(")").split(", ")]
                prob = prob[0] if len(prob) == 1 else prob

            diagnostic_dt = dt.azstats.ess(method=method, prob=prob, sample_dims=sample_dims)
            if grouped:
                diagnostic_dict[diagnostic] = diagnostic_dt.to_stacked_array(
                    "label", sample_dims=[]
                )
            else:
                diagnostic_dict[diagnostic] = diagnostic_dt

        elif "rhat" in diagnostic:
            kwargs = {"sample_dims": sample_dims}
            if diagnostic != "rhat":
                method = diagnostic.split("_", 1)[1]
                kwargs.update({"method": method})

            diagnostic_dt = dt.azstats.rhat(**kwargs)
            if grouped:
                diagnostic_dict[diagnostic] = diagnostic_dt.to_stacked_array(
                    "label", sample_dims=[]
                )
            else:
                diagnostic_dict[diagnostic] = diagnostic_dt
        else:
            warnings.warn(f"{diagnostic} is not recognized as a valid diagnostic")

    if grouped:
        return xr.Dataset(diagnostic_dict)

    return xr.concat(list(diagnostic_dict.values()), dim="diagnostic").assign_coords(
        {"diagnostic": list(diagnostic_dict.keys())}
    )
