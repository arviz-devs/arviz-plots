"""Contain functions for Bayes Factor plotting."""

from copy import copy

import xarray as xr
from arviz_stats.bayes_factor import bayes_factor

from arviz_plots.plots.prior_posterior_plot import plot_prior_posterior
from arviz_plots.plots.utils import add_reference_lines, filter_aes


def plot_bf(
    dt,
    var_names,
    ref_val=0,
    kind=None,
    sample_dims=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    r"""Bayes Factor for comparing hypothesis of two nested models.

    The Bayes factor is estimated by comparing a model (H1) against a model
    in which the parameter of interest has been restricted to be a point-null (H0)
    This computation assumes H0 is a special case of H1. For more details see here
    https://arviz-devs.github.io/EABM/Chapters/Model_comparison.html#savagedickey-ratio

    Parameters
    ----------
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.
    var_names : str, optional
        Variables for which the bayes factor will be computed and the prior and
        posterior will be plotted.
    ref_val : int or float, default 0
        Reference (point-null) value for Bayes factor estimation.
    kind : {"kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
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

        * "ref_line -> passed to :func: `~arviz_plots.visuals.vline`
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
    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    if aes_map is None:
        aes_map = {}
    else:
        aes_map = aes_map.copy()

    bf, _ = bayes_factor(dt, var_names, ref_val, return_ref_vals=True)

    if isinstance(var_names, str):
        var_names = [var_names]
    bf_aes_ds = xr.Dataset(
        {
            var: xr.DataArray(
                None,
                coords={"BF_type": [f"BF01:{bf[var]['BF01']:.2f}"]},
                dims=["BF_type"],
            )
            for var in var_names
        }
    )

    plot_collection = plot_prior_posterior(
        dt,
        var_names=var_names,
        coords=None,
        sample_dims=sample_dims,
        kind=kind,
        plot_collection=plot_collection,
        backend=backend,
        labeller=labeller,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
        pc_kwargs=pc_kwargs,
    )

    plot_collection.update_aes_from_dataset("bf_aes", bf_aes_ds)

    ref_line_kwargs = copy(plot_kwargs.get("ref_line", {}))
    if ref_line_kwargs is False:
        raise ValueError(
            "plot_kwargs['ref_line'] can't be False, use ref_val=False to remove this element"
        )

    if ref_val is not False:
        _, ref_aes, _ = filter_aes(plot_collection, aes_map, "ref_line", "sample")
        if "color" not in ref_aes:
            ref_line_kwargs.setdefault("color", "black")
        if "alpha" not in ref_aes:
            ref_line_kwargs.setdefault("alpha", 0.5)
        add_reference_lines(
            plot_collection, ref_val, aes_map=aes_map, plot_kwargs={"ref_line": ref_line_kwargs}
        )

    if backend == "matplotlib":  ## remove this when we have a better way to handle legends
        plot_collection.add_legend(
            ["__variable__", "BF_type"], loc="upper left", fontsize=10, text_only=True
        )

    return plot_collection
