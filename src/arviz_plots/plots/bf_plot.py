"""Contain functions for Bayes Factor plotting."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import xarray as xr
from arviz_base import rcParams
from arviz_base.validate import validate_dict_argument
from arviz_stats.bayes_factor import bayes_factor

from arviz_plots.plots.prior_posterior_plot import plot_prior_posterior
from arviz_plots.plots.utils import add_lines, filter_aes, get_visual_kwargs
from arviz_plots.visuals import annotate_xy


def plot_bf(
    dt,
    var_names,
    *,
    sample_dims=None,
    ref_val=0,
    kind=None,
    bf_type="BF10",
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "ref_line",
            "ref_value_text",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "dist",
            "ref_line",
            "ref_value_text",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[Literal["dist"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    r"""Bayes Factor for comparing hypothesis of two nested models.

    The Bayes factor is estimated by comparing a model (H1) against a model
    in which the parameter of interest has been restricted to be a point-null (H0).
    This computation assumes H0 is a special case of H1. For more details see
    https://arviz-devs.github.io/EABM/Chapters/Model_comparison.html#savagedickey-ratio

    Parameters
    ----------
    dt : DataTree
        Input data.
    var_names : str, optional
        Variables for which the Bayes factor will be computed and the prior and
        posterior will be plotted.
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    ref_val : int, float, or dict, default 0
        Reference (point-null) value for Bayes factor estimation.
        Can be a single value applied to all variables, or a dict mapping
        variable names to values, e.g. ``{"mu": 0, "tau": 0.5}``.
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    bf_type : {"BF10", "BF01"}, optional
        Whether to annotate the Bayes factor in favor of the alternative (BF10) or the null (BF01).
        Defaults to "BF10".
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
          * "hist" -> passed to :func: `~arviz_plots.visuals.hist`

        * ref_line -> passed to :func: `~arviz_plots.visuals.vline`
        * ref_value_text -> passed to :func:`~arviz_plots.visuals.annotate_xy`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`

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
    Select one variable.

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_bf, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_bf(dt, var_names="mu", kind="hist")

    .. minigallery:: plot_bf
    """
    visuals = validate_dict_argument(visuals, (plot_bf, "visuals"))
    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_bf, "aes_by_visuals"))

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]

    if isinstance(var_names, str):
        var_names = [var_names]

    if isinstance(ref_val, dict):
        ref_vals_list = [ref_val.get(var, 0) for var in var_names]
    else:
        ref_vals_list = [ref_val] * len(var_names)

    bf, ref_densities = bayes_factor(dt, var_names, ref_vals_list, return_ref_vals=True)

    ref_val_ds = xr.Dataset({var: xr.DataArray(rv) for var, rv in zip(var_names, ref_vals_list)})

    keys_to_keep = ("dist", "title")
    plot_collection = plot_prior_posterior(
        dt,
        var_names=var_names,
        coords=None,
        sample_dims=sample_dims,
        kind=kind,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        aes_by_visuals={k: v for k, v in aes_by_visuals.items() if k in keys_to_keep},
        visuals={k: v for k, v in visuals.items() if k in keys_to_keep},
        stats=stats,
        **pc_kwargs,
    )

    ref_line_kwargs = get_visual_kwargs(visuals, "ref_line")
    if ref_line_kwargs is False:
        raise ValueError(
            "visuals['ref_line'] can't be False, use ref_val=False to remove this element"
        )

    if ref_val is not False:
        _, ref_aes, _ = filter_aes(plot_collection, aes_by_visuals, "ref_line", "sample")
        if "color" not in ref_aes:
            ref_line_kwargs.setdefault("color", "B1")
        if "alpha" not in ref_aes:
            ref_line_kwargs.setdefault("alpha", 0.5)
        add_lines(
            plot_collection,
            ref_val,
            aes_by_visuals=aes_by_visuals,
            visuals={"ref_line": ref_line_kwargs},
        )

    ref_value_text_kwargs = get_visual_kwargs(visuals, "ref_value_text")
    if ref_value_text_kwargs is not False:
        _, _, ref_value_text_ignore = filter_aes(
            plot_collection, aes_by_visuals, "ref_value_text", sample_dims
        )
        ref_value_text_kwargs.setdefault("text", lambda bf10: f"BF10={bf10:.2f}")
        ref_value_text_kwargs.setdefault("x", ref_val_ds)
        ref_value_text_kwargs.setdefault("y", ref_densities.min("density_type") * 0.5)
        ref_value_text_kwargs.setdefault("horizontal_align", "left")

        plot_collection.map(
            annotate_xy,
            "ref_value_text",
            data=bf.sel(bf_type=bf_type),
            ignore_aes=ref_value_text_ignore,
            **ref_value_text_kwargs,
        )

    return plot_collection
