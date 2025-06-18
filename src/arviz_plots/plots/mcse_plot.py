"""mcse plot code."""

from collections.abc import Mapping, Sequence
from copy import copy
from importlib import import_module
from typing import Any, Literal

import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import (
    filter_aes,
    get_group,
    process_group_variables_coords,
    set_wrap_layout,
)
from arviz_plots.visuals import (
    annotate_xy,
    labelled_title,
    labelled_x,
    labelled_y,
    line_xy,
    scatter_xy,
    trace_rug,
)


def plot_mcse(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    extra_methods=False,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "mcse",
            "rug",
            "title",
            "xlabel",
            "ylabel",
            "mean",
            "mean_text",
            "sd",
            "sd_text",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "mcse",
            "rug",
            "title",
            "xlabel",
            "ylabel",
            "mean",
            "mean_text",
            "sd",
            "sd_text",
        ],
        Mapping[str, Any] | Literal[False],
    ] = None,
    stats: Mapping[Literal["mcse", "mean", "sd"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    """Plot Monte Carlo standard error.

    The Monte Carlo standard error (mcse) is a measure of the uncertainty associated
    with the estimation of a posterior distribution using Monte Carlo methods.
    See [1]_ for more details.

    Parameters
    ----------
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.
    var_names : str or sequence of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default None
        If None, interpret `var_names` as the real variables names.
        If “like”, interpret `var_names` as substrings of the real variables names.
        If “regex”, interpret `var_names` as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    rug : bool, default False
        Add a `rug plot <https://en.wikipedia.org/wiki/Rug_plot>`_ for a specific subset of values.
    rug_kind : str, default "diverging"
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points : int, default 20
        Number of points for which to plot their quantile/local mcse or number of subsets
        in the evolution plot.
    extra_methods : bool, default False
        Plot mean and sd mcse as horizontal lines.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str or False}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

        By default, no aesthetic mappings are defined. Only when multiple models
        are present a color and x shift is generated to distinguish the data
        coming from the different models.

        When ``mean`` or ``sd`` keys are present in `aes_by_visuals` but ``mean_text``
        or ``sd_text`` are not, the respective ``_text`` key will be added
        with the same values as ``mean`` or ``sd`` ones.

    visuals : mapping of {str : mapping or False}, optional
        Valid keys are:

        * mcse -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * rug -> passed to :func:`~.visuals.trace_rug`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * mean -> passed to :func:`~arviz.plots.visuals.line_xy`
        * mean_text -> passed to :func:`~arviz.plots.visuals.annotate_xy`
        * sd_text -> passed to :func:`~arviz.plots.visuals.annotate_xy`
        * sd -> passed to :func:`~arviz.plots.visuals.line_xy`

    stats : mapping, optional
        Valid keys are:

        * mcse -> passed to mcse, method = 'quantile'
        * mean -> passed to mcse, method='mean'
        * sd -> passed to mcse, method='sd'

    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap`

    Returns
    -------
    PlotCollection

    See Also
    --------
    :ref:`plots_intro` :
        General introduction to batteries-included plotting functions, common use and logic overview

    Examples
    --------
    We can manually map the color to the variable, and have the mapping apply
    to the title too instead of only the mcse markers:

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_mcse, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> non_centered = load_arviz_data('non_centered_eight')
        >>> pc = plot_mcse(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     aes={"color": ["__variable__"]},
        >>>     aes_by_visuals={"title": ["color"]},
        >>> )

    We can add extra methods to plot the mean and standard deviation as lines

    .. plot::
        :context: close-figs

        >>> pc = plot_mcse(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     extra_methods=True,
        >>> )

    Rugs can also be added:

    .. plot::
        :context: close-figs

        >>> pc = plot_mcse(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     rug=True,
        >>> )


    We can also adjust the number of points:

    .. plot::
        :context: close-figs

        >>> pc = plot_mcse(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     n_points=10,
        >>> )

    .. minigallery:: plot_mcse

    References
    ----------
    .. [1] Vehtari et al. *Rank-normalization, folding, and localization: An improved Rhat for
        assmcseing convergence of MCMC*. Bayesian Analysis. 16(2) (2021)
        https://doi.org/10.1214/20-BA1221. arXiv preprint https://arxiv.org/abs/1903.08008
    """
    # initial defaults
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    # mutable inputs
    if visuals is None:
        visuals = {}

    if stats is None:
        stats = {}

    kind = "quantile"

    # processing dt/group/coords/filtering
    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    # ensuring visuals['rug'] is not False
    rug_kwargs = copy(visuals.get("rug", {}))
    if rug_kwargs is False:
        raise ValueError("visuals['rug'] can't be False, use rug=False to remove the rug")

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    # getting backend specific linestyles
    linestyles = plot_bknd.get_default_aes("linestyle", 4, {})
    # and default color
    default_color = plot_bknd.get_default_aes("color", 1, {})[0]

    # set plot collection initialization defaults
    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs.setdefault(
            "cols",
            ["__variable__"]
            + [dim for dim in distribution.dims if dim not in {"model"}.union(sample_dims)],
        )
        if "chain" in distribution:
            pc_kwargs["aes"].setdefault("overlay", ["chain"])
        if "model" in distribution:
            pc_kwargs["aes"].setdefault("color", ["model"])
            n_models = distribution.sizes["model"]
            x_diff = min(1 / n_points / 3, 1 / n_points * n_models / 10)
            pc_kwargs.setdefault("x", np.linspace(-x_diff, x_diff, n_models))
            pc_kwargs["aes"].setdefault("x", ["model"])
        aux_dim_list = [dim for dim in pc_kwargs["cols"] if dim != "__variable__"]

        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, distribution)
        pc_kwargs["figure_kwargs"].setdefault("sharex", True)
        plot_collection = PlotCollection.wrap(
            distribution,
            backend=backend,
            **pc_kwargs,
        )
    else:
        aux_dim_list = list(
            set(
                dim for child in plot_collection.viz.children.values() for dim in child["plot"].dims
            )
        )

    # set plot collection dependent defaults (like aesthetics mappings for each visual)
    if aes_by_visuals is None:
        aes_by_visuals = {}
    else:
        aes_by_visuals = aes_by_visuals.copy()
    aes_by_visuals.setdefault(kind, plot_collection.aes_set.difference({"overlay"}))
    aes_by_visuals.setdefault("rug", {"overlay"})
    if "model" in distribution:
        aes_by_visuals.setdefault("mean", {"color"})
        aes_by_visuals.setdefault("sd", {"color"})
        aes_by_visuals.setdefault("min_mcse", {"color"})
    if "mean" in aes_by_visuals and "mean_text" not in aes_by_visuals:
        aes_by_visuals["mean_text"] = aes_by_visuals["mean"]
    if "sd" in aes_by_visuals and "sd_text" not in aes_by_visuals:
        aes_by_visuals["sd_text"] = aes_by_visuals["sd"]
    if labeller is None:
        labeller = BaseLabeller()

    # compute and add mcse subplots
    mcse_kwargs = copy(visuals.get("mcse", {}))

    if mcse_kwargs is not False:
        mcse_dims, mcse_aes, mcse_ignore = filter_aes(
            plot_collection, aes_by_visuals, kind, sample_dims
        )
        probs = np.linspace(1 / n_points, 1 - 1 / n_points, n_points)

        mcse_y_dataset = xr.concat(
            [
                distribution.azstats.mcse(
                    sample_dims=mcse_dims,
                    method=kind,
                    prob=p,
                    **stats.get("mcse", {}),
                )
                for p in probs
            ],
            dim="mcse_dim",
        )

        xdata_da = xr.DataArray(probs, dims="mcse_dim")
        # broadcasting xdata_da to match shape of each variable in mcse_y_dataset and
        # creating a new dataset from dict of broadcasted xdata
        xdata_dataset = xr.Dataset(
            {var_name: xdata_da.broadcast_like(da) for var_name, da in mcse_y_dataset.items()}
        )
        # concatenating xdata_dataset and mcse_y_dataset along plot_axis
        mcse_dataset = xr.concat([xdata_dataset, mcse_y_dataset], dim="plot_axis").assign_coords(
            plot_axis=["x", "y"]
        )

        if "color" not in mcse_aes:
            mcse_kwargs.setdefault("color", default_color)

        plot_collection.map(
            scatter_xy, "mcse", data=mcse_dataset, ignore_aes=mcse_ignore, **mcse_kwargs
        )

    # plot rug
    sample_stats = get_group(dt, "sample_stats", allow_missing=True)
    if (
        rug
        and sample_stats is not None
        and rug_kind in sample_stats.data_vars
        and np.any(sample_stats[rug_kind])
    ):
        rug_mask = dt.sample_stats[rug_kind]
        _, div_aes, div_ignore = filter_aes(plot_collection, aes_by_visuals, "rug", sample_dims)
        if "color" not in div_aes:
            rug_kwargs.setdefault("color", "black")
        if "marker" not in div_aes:
            rug_kwargs.setdefault("marker", "|")
        if "size" not in div_aes:
            rug_kwargs.setdefault("size", 30)
        div_reduce_dims = [dim for dim in distribution.dims if dim not in aux_dim_list]

        values = distribution.azstats.compute_ranks(dim=sample_dims, relative=True)

        plot_collection.map(
            trace_rug,
            "rug",
            data=values,
            ignore_aes=div_ignore,
            y=distribution.min(div_reduce_dims),
            mask=rug_mask,
            xname=False,
            **rug_kwargs,
        )

    # defining x_range (used for mean, sd)
    x_range = xr.DataArray([0, 1])

    # plot mean and sd and annotate them
    if extra_methods is not False:
        mean_kwargs = copy(visuals.get("mean", {}))
        mean_text_kwargs = copy(visuals.get("mean_text", {}))
        sd_kwargs = copy(visuals.get("sd", {}))
        sd_text_kwargs = copy(visuals.get("sd_text", {}))

        # computing mean_mcse
        mean_dims, mean_aes, mean_ignore = filter_aes(
            plot_collection, aes_by_visuals, "mean", sample_dims
        )
        mean_mcse = None
        if (mean_kwargs is not False) or (mean_text_kwargs is not False):
            mean_mcse = distribution.azstats.mcse(
                sample_dims=mean_dims, method="mean", **stats.get("mean", {})
            )

        # computing sd_mcse
        sd_dims, sd_aes, sd_ignore = filter_aes(plot_collection, aes_by_visuals, "sd", sample_dims)
        sd_mcse = None
        if (sd_kwargs is not False) or (sd_text_kwargs is not False):
            sd_mcse = distribution.azstats.mcse(
                sample_dims=sd_dims, method="sd", **stats.get("sd", {})
            )

        if mean_kwargs is not False:
            # getting 2nd default linestyle for chosen backend and assigning it by default
            if "linestyle" not in mean_aes:
                mean_kwargs.setdefault("linestyle", linestyles[1])

            if "color" not in mean_aes:
                mean_kwargs.setdefault("color", default_color)

            plot_collection.map(
                line_xy,
                "mean",
                data=mean_mcse,
                x=x_range,
                ignore_aes=mean_ignore,
                **mean_kwargs,
            )

        if sd_kwargs is not False:
            if "linestyle" not in sd_aes:
                sd_kwargs.setdefault("linestyle", linestyles[2])

            if "color" not in sd_aes:
                sd_kwargs.setdefault("color", default_color)

            plot_collection.map(
                line_xy, "sd", data=sd_mcse, ignore_aes=sd_ignore, x=x_range, **sd_kwargs
            )

        sd_va_align = None
        mean_va_align = None
        if mean_mcse is not None and sd_mcse is not None:
            sd_va_align = xr.where(mean_mcse < sd_mcse, "bottom", "top")
            mean_va_align = xr.where(mean_mcse < sd_mcse, "top", "bottom")

        if mean_text_kwargs is not False:
            _, mean_text_aes, mean_text_ignore = filter_aes(
                plot_collection, aes_by_visuals, "mean_text", sample_dims
            )

            if "color" not in mean_text_aes:
                mean_text_kwargs.setdefault("color", "black")

            mean_text_kwargs.setdefault("x", 1)
            mean_text_kwargs.setdefault("horizontal_align", "right")

            # pass the mean vertical_align data for vertical alignment setting
            if mean_va_align is not None:
                vertical_align = mean_va_align
            else:
                vertical_align = "bottom"
            mean_text_kwargs.setdefault("vertical_align", vertical_align)

            plot_collection.map(
                annotate_xy,
                "mean_text",
                text="mean",
                data=mean_mcse,
                ignore_aes=mean_text_ignore,
                **mean_text_kwargs,
            )

        if sd_text_kwargs is not False:
            _, sd_text_aes, sd_text_ignore = filter_aes(
                plot_collection, aes_by_visuals, "sd_text", sample_dims
            )

            if "color" not in sd_text_aes:
                sd_text_kwargs.setdefault("color", "black")

            sd_text_kwargs.setdefault("x", 1)
            sd_text_kwargs.setdefault("horizontal_align", "right")

            # pass the sd vertical_align data for vertical alignment setting
            if sd_va_align is not None:
                vertical_align = sd_va_align
            else:
                vertical_align = "top"
            sd_text_kwargs.setdefault("vertical_align", vertical_align)

            plot_collection.map(
                annotate_xy,
                "sd_text",
                text="sd",
                data=sd_mcse,
                ignore_aes=sd_text_ignore,
                **sd_text_kwargs,
            )

    # plot titles for each faceted subplot
    title_kwargs = copy(visuals.get("title", {}))

    if title_kwargs is not False:
        _, title_aes, title_ignore = filter_aes(
            plot_collection, aes_by_visuals, "title", sample_dims
        )
        if "color" not in title_aes:
            title_kwargs.setdefault("color", "black")
        plot_collection.map(
            labelled_title,
            "title",
            ignore_aes=title_ignore,
            subset_info=True,
            labeller=labeller,
            **title_kwargs,
        )

    # plot x and y axis labels
    # Add varnames as x and y labels
    _, labels_aes, labels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "xlabel", sample_dims
    )
    xlabel_kwargs = copy(visuals.get("xlabel", {}))
    if xlabel_kwargs is not False:
        if "color" not in labels_aes:
            xlabel_kwargs.setdefault("color", "black")

        # formatting ylabel and setting xlabel
        xlabel_kwargs.setdefault("text", "Quantile")

        plot_collection.map(
            labelled_x,
            "xlabel",
            ignore_aes=labels_ignore,
            subset_info=True,
            **xlabel_kwargs,
        )

    _, labels_aes, labels_ignore = filter_aes(
        plot_collection, aes_by_visuals, "ylabel", sample_dims
    )
    ylabel_kwargs = copy(visuals.get("ylabel", {}))
    if ylabel_kwargs is not False:
        if "color" not in labels_aes:
            ylabel_kwargs.setdefault("color", "black")

        ylabel_kwargs.setdefault("text", "mcse")

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=labels_ignore,
            subset_info=True,
            **ylabel_kwargs,
        )

    return plot_collection
