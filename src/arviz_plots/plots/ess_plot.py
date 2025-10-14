"""ess plot code."""

from collections.abc import Mapping, Sequence
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
    get_visual_kwargs,
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


def plot_ess(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind="local",
    relative=False,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    extra_methods=False,
    min_ess=400,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal[
            "ess",
            "rug",
            "mean",
            "mean_text",
            "sd",
            "sd_text",
            "min_ess",
            "title",
            "xlabel",
            "ylabel",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "ess",
            "rug",
            "mean",
            "mean_text",
            "sd",
            "sd_text",
            "min_ess",
            "title",
            "xlabel",
            "ylabel",
            "legend",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[Literal["ess", "mean", "sd"], Mapping[str, Any] | xr.Dataset] = None,
    **pc_kwargs,
):
    """Plot effective sample size plots.

    Roughly speaking, the effective sample size of a quantity of interest captures how
    many independent draws contain the same amount of information as the dependent sample
    obtained by the MCMC algorithm. The higher the ESS the better. See [1]_ for more details.

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
    kind : {"local", "quantile"}, default "local"
        Specify the kind of plot:

        * The ``kind="local"`` argument generates the ESS' local efficiency
          for estimating small-interval probability of a desired posterior.
        * The ``kind="quantile"`` argument generates the ESS' local efficiency
          for estimating quantiles of a desired posterior.

    relative : bool, default False
        Show relative ess in plot ``ress = ess / N``.
    rug : bool, default False
        Add a `rug plot <https://en.wikipedia.org/wiki/Rug_plot>`_ for a specific subset of values.
    rug_kind : str, default "diverging"
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points : int, default 20
        Number of points for which to plot their quantile/local ess or number of subsets
        in the evolution plot.
    extra_methods : bool, default False
        Plot mean and sd ESS as horizontal lines.
    min_ess : int, default 400
        Minimum number of ESS desired. If ``relative=True`` the line is plotted at
        ``min_ess / n_samples`` for local and quantile kinds
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

    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * ess -> passed to :func:`~arviz_plots.visuals.scatter_xy`
        * rug -> passed to :func:`~.visuals.trace_rug`
        * mean -> passed to :func:`~arviz.plots.visuals.line_xy`
        * mean_text -> passed to :func:`~arviz.plots.visuals.annotate_xy`
        * sd_text -> passed to :func:`~arviz.plots.visuals.annotate_xy`
        * sd -> passed to :func:`~arviz.plots.visuals.line_xy`
        * min_ess -> passed to :func:`~arviz.plots.visuals.line_xy`
        * title -> passed to :func:`~arviz_plots.visuals.labelled_title`
        * xlabel -> passed to :func:`~arviz_plots.visuals.labelled_x`
        * ylabel -> passed to :func:`~arviz_plots.visuals.labelled_y`
        * legend -> passed to :class:`arviz_plots.PlotCollection.add_legend`

    stats : mapping, optional
        Valid keys are:

        * ess -> passed to ess, method = 'local' or 'quantile' based on `kind`
        * mean -> passed to ess, method='mean'
        * sd -> passed to ess, method='sd'

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
    to the title too instead of only the ess markers:

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_ess, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> non_centered = load_arviz_data('non_centered_eight')
        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     aes={"color": ["__variable__"]},
        >>>     aes_by_visuals={"title": ["color"]},
        >>> )

    We can add extra methods to plot the mean and standard deviation as lines, and adjust
    the minimum ess baseline as well:

    .. plot::
        :context: close-figs

        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     extra_methods=True,
        >>>     min_ess=200,
        >>> )

    Rugs can also be added:

    .. plot::
        :context: close-figs

        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     rug=True,
        >>> )

    Relative ESS can be plotted instead of absolute:

    .. plot::
        :context: close-figs

        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     relative=True,
        >>> )

    We can also adjust the number of points:

    .. plot::
        :context: close-figs

        >>> pc = plot_ess(
        >>>     non_centered,
        >>>     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        >>>     n_points=10,
        >>> )

    .. minigallery:: plot_ess

    References
    ----------
    .. [1] Vehtari et al. *Rank-normalization, folding, and localization: An improved Rhat for
        assessing convergence of MCMC*. Bayesian Analysis. 16(2) (2021)
        https://doi.org/10.1214/20-BA1221. arXiv preprint https://arxiv.org/abs/1903.08008
    """
    # initial defaults
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]

    ylabel = "{}"

    # mutable inputs
    if visuals is None:
        visuals = {}

    if stats is None:
        stats = {}

    # processing dt/group/coords/filtering
    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    # ensuring visuals['rug'] is not False
    rug_kwargs = get_visual_kwargs(visuals, "rug")
    if rug_kwargs is False:
        raise ValueError("visuals['rug'] can't be False, use rug=False to remove the rug")

    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")

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
        pc_kwargs["figure_kwargs"].setdefault("sharey", True)
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
        aes_by_visuals.setdefault("min_ess", {"color"})
    if "mean" in aes_by_visuals and "mean_text" not in aes_by_visuals:
        aes_by_visuals["mean_text"] = aes_by_visuals["mean"]
    if "sd" in aes_by_visuals and "sd_text" not in aes_by_visuals:
        aes_by_visuals["sd_text"] = aes_by_visuals["sd"]
    if labeller is None:
        labeller = BaseLabeller()

    # compute and add ess subplots
    ess_kwargs = get_visual_kwargs(visuals, "ess")

    if ess_kwargs is not False:
        ess_dims, ess_aes, ess_ignore = filter_aes(
            plot_collection, aes_by_visuals, kind, sample_dims
        )
        ylabel = "{}"
        if kind == "local":
            probs = np.linspace(0, 1, n_points, endpoint=False)
        elif kind == "quantile":
            probs = np.linspace(1 / n_points, 1 - 1 / n_points, n_points)
        xdata = probs

        ess_y_dataset = xr.concat(
            [
                distribution.azstats.ess(
                    sample_dims=ess_dims,
                    method=kind,
                    relative=relative,
                    prob=[p, (p + 1 / n_points)] if kind == "local" else p,
                    **stats.get("ess", {}),
                )
                for p in probs
            ],
            dim="ess_dim",
        )

        xdata_da = xr.DataArray(xdata, dims="ess_dim")
        # broadcasting xdata_da to match shape of each variable in ess_y_dataset and
        # creating a new dataset from dict of broadcasted xdata
        xdata_dataset = xr.Dataset(
            {var_name: xdata_da.broadcast_like(da) for var_name, da in ess_y_dataset.items()}
        )
        # concatenating xdata_dataset and ess_y_dataset along plot_axis
        ess_dataset = xr.concat([xdata_dataset, ess_y_dataset], dim="plot_axis").assign_coords(
            plot_axis=["x", "y"]
        )

        if "color" not in ess_aes:
            ess_kwargs.setdefault("color", "C0")

        plot_collection.map(
            scatter_xy, "ess", data=ess_dataset, ignore_aes=ess_ignore, **ess_kwargs
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
            rug_kwargs.setdefault("color", "B1")
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

    # defining x_range (used for mean, sd, minimum ess plotting)
    x_range = xr.DataArray([0, 1])

    # plot mean and sd and annotate them
    if extra_methods is not False:
        mean_kwargs = get_visual_kwargs(visuals, "mean")
        mean_text_kwargs = get_visual_kwargs(visuals, "mean_text")
        sd_kwargs = get_visual_kwargs(visuals, "sd")
        sd_text_kwargs = get_visual_kwargs(visuals, "sd_text")

        # computing mean_ess
        mean_dims, mean_aes, mean_ignore = filter_aes(
            plot_collection, aes_by_visuals, "mean", sample_dims
        )
        mean_ess = None
        if (mean_kwargs is not False) or (mean_text_kwargs is not False):
            mean_ess = distribution.azstats.ess(
                sample_dims=mean_dims, method="mean", relative=relative, **stats.get("mean", {})
            )

        # computing sd_ess
        sd_dims, sd_aes, sd_ignore = filter_aes(plot_collection, aes_by_visuals, "sd", sample_dims)
        sd_ess = None
        if (sd_kwargs is not False) or (sd_text_kwargs is not False):
            sd_ess = distribution.azstats.ess(
                sample_dims=sd_dims, method="sd", relative=relative, **stats.get("sd", {})
            )

        if mean_kwargs is not False:
            # getting 2nd default linestyle for chosen backend and assigning it by default
            if "linestyle" not in mean_aes:
                mean_kwargs.setdefault("linestyle", "C1")

            if "color" not in mean_aes:
                mean_kwargs.setdefault("color", "C0")

            plot_collection.map(
                line_xy,
                "mean",
                data=mean_ess,
                x=x_range,
                ignore_aes=mean_ignore,
                **mean_kwargs,
            )

        if sd_kwargs is not False:
            if "linestyle" not in sd_aes:
                sd_kwargs.setdefault("linestyle", "C2")

            if "color" not in sd_aes:
                sd_kwargs.setdefault("color", "C0")

            plot_collection.map(
                line_xy, "sd", data=sd_ess, ignore_aes=sd_ignore, x=x_range, **sd_kwargs
            )

        sd_va_align = None
        mean_va_align = None
        if mean_ess is not None and sd_ess is not None:
            sd_va_align = xr.where(mean_ess < sd_ess, "bottom", "top")
            mean_va_align = xr.where(mean_ess < sd_ess, "top", "bottom")

        if mean_text_kwargs is not False:
            _, mean_text_aes, mean_text_ignore = filter_aes(
                plot_collection, aes_by_visuals, "mean_text", sample_dims
            )

            if "color" not in mean_text_aes:
                mean_text_kwargs.setdefault("color", "B1")

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
                data=mean_ess,
                ignore_aes=mean_text_ignore,
                **mean_text_kwargs,
            )

        if sd_text_kwargs is not False:
            _, sd_text_aes, sd_text_ignore = filter_aes(
                plot_collection, aes_by_visuals, "sd_text", sample_dims
            )

            if "color" not in sd_text_aes:
                sd_text_kwargs.setdefault("color", "B1")

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
                data=sd_ess,
                ignore_aes=sd_text_ignore,
                **sd_text_kwargs,
            )

    # plot minimum ess
    min_ess_kwargs = get_visual_kwargs(visuals, "min_ess")

    if min_ess_kwargs is not False:
        _, min_ess_aes, min_ess_ignore = filter_aes(
            plot_collection, aes_by_visuals, "min_ess", sample_dims
        )

        if relative:
            min_ess = min_ess / n_points

        if "linestyle" not in min_ess_aes:
            min_ess_kwargs.setdefault("linestyle", "C3")

        if "color" not in min_ess_aes:
            min_ess_kwargs.setdefault("color", "B2")

        plot_collection.map(
            line_xy,
            "min_ess",
            data=distribution,
            ignore_aes=min_ess_ignore,
            x=x_range,
            y=min_ess,
            **min_ess_kwargs,
        )

    # plot titles for each faceted subplot
    title_kwargs = get_visual_kwargs(visuals, "title")

    if title_kwargs is not False:
        _, title_aes, title_ignore = filter_aes(
            plot_collection, aes_by_visuals, "title", sample_dims
        )
        if "color" not in title_aes:
            title_kwargs.setdefault("color", "B1")
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
    xlabel_kwargs = get_visual_kwargs(visuals, "xlabel")
    if xlabel_kwargs is not False:
        if "color" not in labels_aes:
            xlabel_kwargs.setdefault("color", "B1")

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
    ylabel_kwargs = get_visual_kwargs(visuals, "ylabel")
    if ylabel_kwargs is not False:
        if "color" not in labels_aes:
            ylabel_kwargs.setdefault("color", "B1")

        if relative is not False:
            ylabel_text = ylabel.format("Relative ESS")
        else:
            ylabel_text = ylabel.format("ESS")
        ylabel_kwargs.setdefault("text", ylabel_text)

        plot_collection.map(
            labelled_y,
            "ylabel",
            ignore_aes=labels_ignore,
            subset_info=True,
            **ylabel_kwargs,
        )

    # legend
    if "model" in distribution:
        legend_kwargs = get_visual_kwargs(visuals, "legend")
        if legend_kwargs is not False:
            legend_kwargs.setdefault("dim", ["model"])
            plot_collection.add_legend(**legend_kwargs)

    return plot_collection
