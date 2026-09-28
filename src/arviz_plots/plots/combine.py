"""Elements to combine multiple batteries-included plots into a single figure."""

import itertools
from importlib import import_module

import numpy as np
from arviz_base import rcParams
from xarray import Dataset

from arviz_plots import PlotCollection
from arviz_plots.plot_collection import backend_from_object
from arviz_plots.plots.utils import (
    process_group_variables_coords,
    set_grid_layout,
    set_wrap_layout,
)


def render(da, target, **kwargs):
    """Render visual descriptions from the none backend with a plotting backend."""
    backend = backend_from_object(target, return_module=False)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    visuals = da.item().copy()
    plot_fun_name = visuals.pop("function")
    return getattr(plot_backend, plot_fun_name)(target=target, **{**visuals, **kwargs})


def combine_plots(
    dt=None,
    plots=None,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    expand="wrap",
    col_wrap=4,
    plot_names=None,
    backend=None,
    **pc_kwargs,
):
    """Arrange multiple batteries-included plots in a customizable layout.

    Parameters
    ----------
    dt : DataTree of dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.

        Note that not all batteries included functions accept dictionary input, so it will
        only work when all plotting functions requested in `plots` are compatible with it.
    plots : list of tuple
        The plotting functions and the arguments to be used for each plot.
        - a 2-tuple ``(callable, kwargs)``. The `callable` is applied to the datatree
        passed by the `dt` argument.
        - a 3-tuple ``(callable, data, kwargs)``. The `callable` is applied to
        `data` instead of the datatree `dt`, so different plots can use different data.

        `kwargs` is passed as keyword arguments to `callable`. Any of `group`,
        `var_names`, `filter_vars`, `coords`, or `sample_dims` given here take
        precedence, for the associated callable, over the same-named argument
        passed to `combine_plots` itself, for that one subplot only.
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
    expand : {"wrap", "column", "row"}, default "wrap"
        Layout used to combine the different plotting functions.

        - "wrap": all plots are wrapped in a layout with `col_wrap` columns.
        - "column": each tuple in `plots`, is mapped to its own column, with
          the generated plots stacked vertically as rows.
        - "row": each tuple in `plots`, is mapped to its own row, with
          the generated plots laid out horizontally as columns.

        When using "column" or "row" plotting functions expanding to different
        numbers of plot  may leave the remaining grid cells blank.
    col_wrap : int, optional
        Number of columns to wrap the subplots into. Only used when
        `expand="wrap"`, ignored otherwise. Defaults to 4.
    plot_names : list of str, optional
        List of the same length as `plots` with the plot names to use as coordinate values
        in the returned :class:`~arviz_plots.PlotCollection`.
    backend : {"matplotlib", "bokeh", "plotly"}, optional
        Plotting backend to use. Defaults to ``rcParams["plot.backend"]``.
    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.wrap` when `expand="wrap"`,
        or to :class:`arviz_plots.PlotCollection.grid` when `expand` is
        "column" or "row".

    Returns
    -------
    PlotCollection

    Examples
    --------
    Customize the names of the plots in the returned :class:`PlotCollection`

    .. plot::
        :context: close-figs

        >>> import arviz_plots as azp
        >>> azp.style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> rugby = load_arviz_data('rugby')
        >>> pc = azp.combine_plots(
        >>>     rugby,
        >>>     plots=[
        >>>         (azp.plot_ppc_pit, {}),
        >>>         (azp.plot_ppc_rootogram, {}),
        >>>     ],
        >>>     group="posterior_predictive",
        >>>     plot_names=["pit", "rootogram"],
        >>> )

    Now if we inspect the ``pc.viz`` attribute, we can see it has an ``item`` dimension
    with one entry per variable of each plot:

    .. plot::
        :context: close-figs

        >>> pc.viz

    Compare, side by side, posterior marginal distributions of different models.

    .. plot::
        :context: close-figs

        crabs_pois = azb.load_arviz_data('crabs_poisson')
        crabs_hurdle = azb.load_arviz_data('crabs_hurdle_nb')
        azp.combine_plots(
            plots=[
            (azp.plot_dist, crabs_pois, {"visuals": {"title": {"text": "Poisson"}}}),
            (azp.plot_dist, crabs_hurdle, {"visuals": {"title": {"text": "Hurdle NB"}}}),

                ],
            group="posterior",
            var_names=["C(color)", "width"],
            expand="column",
        );

    .. minigallery:: combine_plots
    """
    if plots is None:
        raise ValueError("`plots` is required.")

    if expand not in {"wrap", "column", "row"}:
        raise ValueError(f"`expand` must be 'wrap', 'column' or 'row' but got '{expand}'")

    def _validate(elem):
        if len(elem) == 2:
            func, kwargs = elem
            if dt is None:
                raise ValueError("`dt` must be provided for 2-tuple entries.")
            return func, dt, kwargs
        if len(elem) == 3:
            return elem
        raise ValueError("Each `plots` entry must be a 2- or 3-tuple.")

    validated = [_validate(elem) for elem in plots]
    if not validated:
        raise ValueError("`plots` must contain at least one plot.")

    if plot_names is None:
        plot_names = [
            getattr(func, "__name__", "plot") + f"_{idx:02d}"
            for idx, (func, _, _) in enumerate(validated)
        ]
    elif len(set(plot_names)) != len(plot_names):
        raise ValueError("`plot_names` must be unique.")
    if backend is None:
        backend = rcParams["plot.backend"]

    original_pc_kwargs = pc_kwargs.copy()

    pcs = []
    per_subplot_items = []

    for plot, data_i, kwargs in validated:
        kwargs_i = original_pc_kwargs | kwargs
        group_i = kwargs.get("group", group)
        var_names_i = kwargs.get("var_names", var_names)
        filter_vars_i = kwargs.get("filter_vars", filter_vars)
        coords_i = kwargs.get("coords", coords)
        sample_dims_i = kwargs.get("sample_dims", sample_dims)
        kwargs_i.update(
            group=group_i,
            var_names=var_names_i,
            filter_vars=filter_vars_i,
            coords=coords_i,
            sample_dims=sample_dims_i,
        )

        distribution_i = process_group_variables_coords(
            data_i,
            group=group_i,
            var_names=var_names_i,
            filter_vars=filter_vars_i,
            coords=coords_i,
        )
        if "__variable__" in distribution_i.coords:
            variables = list(distribution_i.coords["__variable__"].values)
        else:
            variables = list(distribution_i.data_vars)

        pc_i = plot(data_i, backend="none", **kwargs_i)

        subplot_items = []
        for variable in variables:
            da = pc_i.viz["row_index"].dataset[variable]
            extra_dims = list(da.dims)
            if not extra_dims:
                subplot_items.append((variable, {}, ""))
                continue
            coord_lists = [da.coords[d].values.tolist() for d in extra_dims]
            for combo in itertools.product(*coord_lists):
                sel = dict(zip(extra_dims, combo))
                suffix = ",".join(f"{v}" for v in combo)
                subplot_items.append((variable, sel, suffix))

        per_subplot_items.append(subplot_items)
        pcs.append(pc_i)

    items = [
        f"{name}::{variable}" + (f"[{suffix}]" if suffix else "")
        for name, subplot_items in zip(plot_names, per_subplot_items)
        for variable, _, suffix in subplot_items
    ]
    n_items = len(items)
    if len(set(items)) != n_items:
        raise ValueError("Duplicated subplot/variable labels; pass distinct `plot_names`.")

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    pc_kwargs = original_pc_kwargs.copy()
    pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
    if expand == "wrap":
        layout = Dataset({"_": (("item",), np.arange(n_items))}, coords={"item": items})
        pc_kwargs.setdefault("cols", ["item"])
        pc_kwargs["col_wrap"] = col_wrap
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, layout)
        pc = PlotCollection.wrap(layout, backend=backend, **pc_kwargs)
    else:
        counts = [len(subplot_items) for subplot_items in per_subplot_items]
        max_items = max(counts)
        if expand == "column":
            layout_dims = ("item", "column")
            layout_shape = (max_items, len(plot_names))
            layout_coords = {"item": np.arange(max_items), "column": plot_names}
            grid_rows, grid_cols = ["item"], ["column"]
        else:
            layout_dims = ("row", "item")
            layout_shape = (len(plot_names), max_items)
            layout_coords = {"row": plot_names, "item": np.arange(max_items)}
            grid_rows, grid_cols = ["row"], ["item"]
        layout = Dataset(
            {"_": (layout_dims, np.arange(int(np.prod(layout_shape))).reshape(layout_shape))},
            coords=layout_coords,
        )
        pc_kwargs.setdefault("rows", grid_rows)
        pc_kwargs.setdefault("cols", grid_cols)
        pc_kwargs = set_grid_layout(pc_kwargs, plot_bknd, layout)
        pc = PlotCollection.grid(layout, backend=backend, **pc_kwargs)

    for name, subplot_items, pc_i in zip(plot_names, per_subplot_items, pcs):
        for sub_idx, (variable, sel, suffix) in enumerate(subplot_items):
            item_label = f"{name}::{variable}" + (f"[{suffix}]" if suffix else "")
            pc.coords = None
            pc.aes = pc_i.aes
            if expand == "wrap":
                pc.coords = {"item": item_label}
            else:
                pc.coords = {expand: name, "item": sub_idx}
            for viz_group, ds in pc_i.viz.children.items():
                if viz_group in {"plot", "col_index", "row_index"}:
                    continue
                if variable not in ds.dataset.data_vars:
                    continue
                data_i = ds.dataset[[variable]]
                selection = {k: v for k, v in sel.items() if k in data_i.dims}
                if selection:
                    data_i = data_i.sel(selection)
                attrs = ds.attrs
                pc.map(
                    render,
                    f"{viz_group}_{item_label}",
                    data=data_i,
                    ignore_aes=attrs.get("ignore_aes", frozenset()),
                )

    pc.coords = None
    return pc
