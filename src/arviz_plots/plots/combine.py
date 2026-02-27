"""Elements to combine multiple batteries-included plots into a single figure."""
from importlib import import_module

from arviz_base import rcParams

from arviz_plots import PlotCollection
from arviz_plots.plot_collection import backend_from_object
from arviz_plots.plots.utils import process_group_variables_coords, set_grid_layout


def render(da, target, **kwargs):
    """Render visual descriptions from the none backend with a plotting backend."""
    backend = backend_from_object(target, return_module=False)
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    visuals = da.item().copy()
    plot_fun_name = visuals.pop("function")
    return getattr(plot_backend, plot_fun_name)(target=target, **{**visuals, **kwargs})


def combine_plots(
    dt,
    plots,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    expand="column",
    plot_names=None,
    backend=None,
    **pc_kwargs,
):
    """Arrange multiple batteries-included plots in a customizable column or row layout.

    Parameters
    ----------
    dt : DataTree of dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.

        Note that not all batteries included functions accept dictionary input, so it will
        only work when all plotting functions requested in `plots` are compatible with it.
    plots : list of tuple of (callable, mapping)
        List of all the plotting functions to be combined. Each element in this list
        is a tuple with two elements. The first is the function to be called, the second
        is a dictionary with any keyword arguments that should be used when calling that function.
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
    expand : {"column", "row"}, default "column"
        How to combine the different plotting functions. If "column", each plotting function
        will be added as a new column, if "row" it will be a new row instead.
    plot_names : list of str, optional
        List of the same length as `plots` with the plot names to use as coordinate values
        in the returned :class:`~arviz_plots.PlotCollection`.
    backend : {"matplotlib", "bokeh", "plotly"}, optional
        Plotting backend to use. Defaults to ``rcParams["plot.backend"]``.
    **pc_kwargs
        Passed to :class:`arviz_plots.PlotCollection.grid`

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

    Now if we inspect the ``pc.viz`` attribute, we can see it has a ``column`` dimension
    with the requested coordinate values:

    .. plot::
        :context: close-figs

        >>> pc.viz

    .. minigallery:: combine_plots
    """
    if plot_names is None:
        plot_names = [
            getattr(elem[0], "__name__") + f"_{idx:02d}" for idx, elem in enumerate(plots)
        ]
    if sample_dims is None:
        sample_dims = rcParams["data.sample_dims"]
    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if backend is None:
        backend = rcParams["plot.backend"]

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    facet_dims = ["__variable__"] + (
        []
        if "predictive" in group
        else [dim for dim in distribution.dims if dim not in sample_dims]
    )

    pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
    if expand == "column":
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs.setdefault("rows", facet_dims)
        expand_kwargs = {"column": len(plots)}
    elif expand == "row":
        pc_kwargs.setdefault("cols", facet_dims)
        pc_kwargs.setdefault("rows", ["row"])
        expand_kwargs = {"row": len(plots)}
    else:
        raise ValueError(f"`expand` must be 'row' or 'column' but got '{expand}'")
    distribution = distribution.expand_dims(**expand_kwargs).assign_coords({expand: plot_names})

    plot_bknd = import_module(f".backend.{backend}", package="arviz_plots")
    pc_kwargs = set_grid_layout(pc_kwargs, plot_bknd, distribution)

    pc = PlotCollection.grid(
        distribution,
        backend=backend,
        **pc_kwargs,
    )

    for name, (plot, kwargs) in zip(plot_names, plots):
        pc_i = plot(
            dt,
            backend="none",
            group=group,
            var_names=var_names,
            filter_vars=filter_vars,
            coords=coords,
            sample_dims=sample_dims,
            **kwargs,
        )
        pc.coords = None
        pc.aes = pc_i.aes
        pc.coords = {expand: name}
        for viz_group, ds in pc_i.viz.children.items():
            if viz_group in {"plot", "row_index", "col_index"}:
                continue
            attrs = ds.attrs
            pc.map(
                render,
                f"{viz_group}_{name}",
                data=ds.dataset,
                ignore_aes=attrs.get("ignore_aes", frozenset()),
            )
    pc.coords = None
    # TODO: at some point all `pc_i.aes` objects should be merged
    # and stored into the `pc.aes` attribute

    return pc
