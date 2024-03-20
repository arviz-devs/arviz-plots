"""Forest plot code."""
import arviz_stats  # pylint: disable=unused-import
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller

from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, process_group_variables_coords
from arviz_plots.visuals import annotate_label, line_x, remove_axis, scatter_x, xticks


def plot_forest(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    combined=False,
    point_estimate=None,
    ci_kind=None,
    ci_prob=None,
    labels=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_map=None,
    plot_kwargs=None,
    stats_kwargs=None,
    pc_kwargs=None,
):
    """Plot 1D marginal credible intervals in a single plot.

    For a general introduction to batteries included functions like this one and common
    usage examples see :ref:`plots_intro`

    Parameters
    ----------
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.
    var_names: str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars: {None, “like”, “regex”}, default=None
        If None, interpret var_names as the real variables names.
        If “like”, interpret var_names as substrings of the real variables names.
        If “regex”, interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
    sample_dims : iterable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    combined : bool, False
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to ``rcParams["plot.point_estimate"]``
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible interval.
        Defaults to ``rcParams["stats.ci_prob"]``
    labels : iterable of str, optional
        Iterable with the dimensions to be labelled in the plot. By default all dimensions.
        It can include the special "__variable__" indicator, and does so by default.
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh"}, optional
    labeller : labeller, optional
    aes_map : mapping of {str : iterable of str}, optional
        Mapping of artists to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `plot_kwargs`.

        Defaults to only mapping properties to the density representation.
        And when "point_estimate" key is provided but "point_estimate_text" isn't,
        the values assigned to the first are also used for the second.
    plot_kwargs : mapping, optional
        Valid keys are:

        * credible_interval -> passed to :func:`~arviz_plots.visuals.line_x`
        * point_estimate -> passed to :func:`~arviz_plots.visuals.scatter_x`

    stats_kwargs : mapping, optional
        Valid keys are:

        * credible_interval -> passed to eti or hdi
        * point_estimate -> passed to mean, median or mode

    pc_kwargs : mapping
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection

    Notes
    -----
    The separation between variables and all its coordinate values is set to 1.
    The only two exceptions to this are the dimensions named "chain" and "model"
    in case they are present, which get a smaller spacing to give a sense of
    grouping among visual elements that only differ on their chain or model id.

    Examples
    --------
    Default forest plot for a single model:

    .. plot::
        :context: close-figs

        >>> from arviz_plots import plot_forest
        >>> from arviz_base import load_arviz_data
        >>> centered = load_arviz_data('centered_eight')
        >>> non_centered = load_arviz_data('non_centered_eight')
        >>> pc = plot_forest(centered)

    Default forest plot for multiple models:

    .. plot::
        :context: close-figs

        >>> pc = plot_forest({"centered": centered, "non centered": non_centered})
        >>> pc.add_legend("model")

    Single model forest plot with color mapped to the variable:

    .. plot::
        :context: close-figs

        >>> pc = plot_forest(
        >>>     non_centered,
        >>>     pc_kwargs={"aes": {"color": ["__variable__"]}}
        >>> )

    """
    if ci_kind not in ["hdi", "eti", None]:
        raise ValueError("ci_kind must be either 'hdi' or 'eti'")

    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]
    if ci_kind is None:
        ci_kind = rcParams["stats.ci_kind"] if "stats.ci_kind" in rcParams else "eti"
    if point_estimate is None:
        point_estimate = rcParams["plot.point_estimate"]
    if plot_kwargs is None:
        plot_kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()

    if stats_kwargs is None:
        stats_kwargs = {}

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    labellable_dims = ["__variable__"] + [
        dim for dim in distribution.dims if (dim not in sample_dims) and (dim != "model")
    ]
    if labels is None:
        labels = labellable_dims

    if plot_collection is None:
        if backend is None:
            backend = rcParams["plot.backend"]
        pc_kwargs.setdefault("cols", ["__column__"])
        pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
        pc_kwargs["plot_grid_kws"].setdefault("sharey", True)
        pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
        pc_kwargs["aes"].setdefault("y", labellable_dims)
        pc_kwargs["aes"].setdefault("overlay", labellable_dims)
        if "model" in distribution.dims:
            pc_kwargs["aes"].setdefault("color", ["model"])
        plot_collection = PlotCollection.grid(
            distribution.expand_dims(__column__=2).assign_coords(__column__=["labels", "forest"]),
            backend=backend,
            **pc_kwargs,
        )
    if plot_collection.aes is None:
        plot_collection.generate_aes_dt()

    # fine tune y position for model and chain
    aes_dt = plot_collection.aes
    if combined and "model" in distribution.dims:
        for child in aes_dt.children.values():
            child["y"] = child["y"] + xr.DataArray(
                np.linspace(-0.2, 0.2, distribution.sizes["model"]),
                coords={"model": distribution.model},
            )
    elif ("model" in distribution.dims) and ("chain" in distribution.dims):
        for child in aes_dt.children.values():
            model_spacing = xr.DataArray(
                np.linspace(-0.2, 0.2, distribution.sizes["model"]),
                coords={"model": distribution.model},
            )
            chain_lim = 0.4 * (model_spacing[1] - model_spacing[0])
            chain_spacing = xr.DataArray(
                np.linspace(-chain_lim, chain_lim, distribution.sizes["chain"]),
                coords={"chain": distribution.chain},
            )
            child["y"] = child["y"] + model_spacing + chain_spacing
    elif "chain" in distribution.dims:
        for child in aes_dt.children.values():
            child["y"] = child["y"] + xr.DataArray(
                np.linspace(-0.2, 0.2, distribution.sizes["chain"]),
                coords={"chain": distribution.chain},
            )
    plot_collection.aes = aes_dt

    if aes_map is None:
        aes_map = {
            "credible_interval": plot_collection.aes_set,
            "point_estimate": plot_collection.aes_set,
        }
    else:
        aes_map = aes_map.copy()
    aes_map["labels"] = {"overlay"}.union(aes_map.get("labels", {}))
    if "point_estimate" in aes_map and "point_estimate_text" not in aes_map:
        aes_map["point_estimate_text"] = aes_map["point_estimate"]
    if labeller is None:
        labeller = BaseLabeller()

    # credible interval
    ci_dims, ci_aes, ci_ignore = filter_aes(
        plot_collection, aes_map, "credible_interval", sample_dims
    )
    if ci_kind == "eti":
        ci = distribution.azstats.eti(
            prob=ci_prob, dims=ci_dims, **stats_kwargs.get("credible_interval", {})
        )
    elif ci_kind == "hdi":
        ci = distribution.azstats.hdi(
            prob=ci_prob, dims=ci_dims, **stats_kwargs.get("credible_interval", {})
        )

    ci_kwargs = plot_kwargs.get("credible_interval", {}).copy()
    if "color" not in ci_aes:
        ci_kwargs.setdefault("color", "gray")
    plot_collection.map(
        line_x,
        "credible_interval",
        data=ci,
        ignore_aes=ci_ignore,
        coords={"__column__": "forest"},
        **ci_kwargs,
    )

    # point estimate
    pe_dims, pe_aes, pe_ignore = filter_aes(plot_collection, aes_map, "point_estimate", sample_dims)
    if point_estimate == "median":
        point = distribution.median(dim=pe_dims, **stats_kwargs.get("point_estimate", {}))
    elif point_estimate == "mean":
        point = distribution.mean(dim=pe_dims, **stats_kwargs.get("point_estimate", {}))
    else:
        raise NotImplementedError("coming soon")

    pe_kwargs = plot_kwargs.get("point_estimate", {}).copy()
    if "color" not in pe_aes:
        pe_kwargs.setdefault("color", "black")
    plot_collection.map(
        scatter_x,
        "point_estimate",
        data=point,
        ignore_aes=pe_ignore,
        coords={"__column__": "forest"},
        **pe_kwargs,
    )

    _, lab_aes, lab_ignore = filter_aes(plot_collection, aes_map, "labels", sample_dims)
    lab_kwargs = plot_kwargs.get("labels", {}).copy()
    if "color" not in lab_aes:
        lab_kwargs.setdefault("color", "black")
    y_ds = xr.Dataset({key: values["y"] for key, values in plot_collection.aes.children.items()})
    cumulative_label = ["__variable__"]
    for x, label in enumerate(labels):
        if label == "__variable__":
            y = (y_ds.max() + y_ds.min()) / 2
        else:
            cumulative_label.append(label)
            reduce_dims = [dim for dim in y_ds.dims if dim not in cumulative_label]
            y = (y_ds.max(reduce_dims) + y_ds.min(reduce_dims)) / 2
        plot_collection.map(
            annotate_label,
            f"{label.strip('_')}_label",
            data=y,
            x=x,
            dim=None if label == "__variable__" else label,
            subset_info=True,
            coords={"__column__": "labels"},
            ignore_aes=lab_ignore,
            **lab_kwargs,
        )
    plot_collection.map(
        xticks,
        store_artist=False,
        loop_data="plots",
        ignore_aes=plot_collection.aes_set,
        ticks=np.arange(len(labels)),
        labels=[label.strip("_") for label in labels],
        coords={"__column__": "labels"},
    )

    plot_collection.map(
        remove_axis, store_artist=False, axis="y", ignore_aes=plot_collection.aes_set
    )

    return plot_collection
