"""Pair plot code."""
from copy import copy
from importlib import import_module

import numpy as np
from arviz_base import rcParams, xarray_sel_iter
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_matrix import PlotMatrix
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    process_group_variables_coords,
    set_grid_layout,
)
from arviz_plots.visuals import label_plot, scatter_couple


def plot_pair(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    plot_matrix=None,
    backend=None,
    labeller=None,
    aes_by_visuals=None,
    visuals=None,
    **pc_kwargs,
):
    """Plot all variables against each other in the dataset.

    Parameters
    ----------
    dt : DataTree
        Input data
    var_names: str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars: {None, “like”, “regex”}, default None
        If None (default), interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to use for plotting. Defaults to "posterior".
    coords : mapping, optional
        Coordinates to use for plotting.
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    plot_matrix : PlotMatrix, optional
    backend : {"matplotlib", "bokeh", "plotly", "none"}, optional
        Plotting backend to use. Defaults to ``rcParams["plot.backend"]``
    labeller : labeller, optional
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_matrix`
        when plotted. Valid keys are the same as for `visuals`.
        By default, there are no aesthetic mappings at all
    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * scatter -> passed to :func:`~.visuals.scatter_couple`
        * divergence -> passed to :func:`~.visuals.scatter_couple`. Defaults to False.
        * label -> :func:`~.visuals.label_plot`

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotMatrix`


    Returns
    -------
    PlotMatrix

    Examples
    --------
    Default plot_pair

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_pair, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> dt = load_arviz_data('centered_eight')
        >>> plot_pair(
        >>>     dt,
        >>>     var_names=["mu", "theta"],
        >>>     coords={"school": "Choate"},
        >>>     visuals={"divergence": True},
        >>> )

    .. minigallery:: plot_pair

    """
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if visuals is None:
        visuals = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()

    if backend is None:
        if plot_matrix is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_matrix.backend

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

    if plot_matrix is None:
        pc_kwargs.setdefault(
            "facet_dims",
            ["__variable__"] + [dim for dim in distribution.dims if dim not in sample_dims],
        )
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pairs = tuple(
            xarray_sel_iter(
                distribution, skip_dims={dim for dim in distribution.dims if dim in sample_dims}
            )
        )
        n_pairs = len(pairs)
        pc_kwargs = set_grid_layout(
            pc_kwargs, plot_bknd, distribution, num_rows=n_pairs, num_cols=n_pairs
        )
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        pc_kwargs["figure_kwargs"].setdefault("sharey", True)
        plot_matrix = PlotMatrix(
            distribution,
            backend=backend,
            **pc_kwargs,
        )

    # scatter

    aes_by_visuals["scatter"] = {"overlay"}.union(aes_by_visuals.get("scatter", {}))
    scatter_kwargs = copy(visuals.get("scatter", {}))
    if scatter_kwargs is not False:
        _, scatter_aes, scatter_ignore = filter_aes(
            plot_matrix, aes_by_visuals, "scatter", sample_dims
        )

        if "color" not in scatter_aes:
            colors = plot_bknd.get_default_aes("color", 1, {})
            scatter_kwargs.setdefault("color", colors[0])

        if "width" not in scatter_aes:
            scatter_kwargs.setdefault("width", 0)

        if "alpha" not in scatter_aes:
            scatter_kwargs.setdefault("alpha", 0.5)

        plot_matrix.map_triangle(
            scatter_couple,
            "scatter",
            data=distribution,
            ignore_aes=scatter_ignore,
            **scatter_kwargs,
        )

    # divergence

    aes_by_visuals["divergence"] = {"overlay"}.union(aes_by_visuals.get("divergence", {}))
    div_kwargs = copy(visuals.get("divergence", False))
    if div_kwargs is True:
        div_kwargs = {}
    sample_stats = get_group(dt, "sample_stats", allow_missing=True)
    if (
        div_kwargs is not False
        and sample_stats is not None
        and "diverging" in sample_stats.data_vars
        and np.any(sample_stats.diverging)
    ):
        divergence_mask = dt.sample_stats.diverging
        _, div_aes, div_ignore = filter_aes(plot_matrix, aes_by_visuals, "divergence", sample_dims)
        if "color" not in div_aes:
            div_kwargs.setdefault("color", "black")
        if "alpha" not in div_aes:
            div_kwargs.setdefault("alpha", 0.5)
        plot_matrix.map_triangle(
            scatter_couple,
            "divergence",
            data=distribution,
            ignore_aes=div_ignore,
            mask=divergence_mask,
            **div_kwargs,
        )

    if labeller is None:
        labeller = BaseLabeller()

    # diagonal label of plots

    label_kwargs = copy(visuals.get("label", {}))

    if label_kwargs is not False:
        _, _, xlabel_ignore = filter_aes(plot_matrix, aes_by_visuals, "label", sample_dims)
        plot_matrix.map(
            label_plot,
            "label",
            subset_info=True,
            ignore_aes=xlabel_ignore,
            labeller=labeller,
            **label_kwargs,
        )

    return plot_matrix
