"""Elements to combine multiple batteries-included plots into a single chart."""
import re
from importlib import import_module

import xarray as xr
from arviz_base import rcParams

from arviz_plots import PlotCollection
from arviz_plots.plots.utils import process_group_variables_coords, set_figure_layout

def get_valid_arg(key, value, backend):
    """Convert none backend aesthetic argument indicator to a valid value for the given backend.

    Parameters
    ----------
    key : str
        The keyword part of the :ref:`backend-interface-arguments` for which `value` should
        be valid.
    value : any
        The current value for `key`. It might be an indicator from the none backend such as
        "color_0" or "linestyle_3" which gets processed or something else in which case
        it is assumed to be a valid argument already and returned as is.
    backend : str
        The backend for which `value` should be valid.

    Returns
    -------
    valid_value : any
    """
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    key_matcher = "color" if key in {"facecolor", "edgecolor"} else key
    if isinstance(value, str):
        match = re.match(key_matcher +"_([0-9]+)", value)
        if match:
            index = int(match.groups()[0])
            return plot_backend.get_default_aes(key, index+1)[index]
    return value


def backendize_kwargs(kwargs, backend):
    """Process the artist description dictionary from the none backend to valid kwargs."""
    return {key: get_valid_arg(key, value, backend) for key, value in kwargs.items() if key != "function"}

def render(da, target, backend, **kwargs):
    """Render artist descriptions from the none backend with a plotting backend."""
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    plot_kwargs = da.item()
    plot_fun_name = plot_kwargs["function"]
    plot_kwargs = backendize_kwargs(plot_kwargs, backend)
    kwargs = backendize_kwargs(kwargs, backend)
    return getattr(plot_backend, plot_fun_name)(target=target, **{**plot_kwargs, **kwargs})

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
    pc_kwargs=None,
):
    """Create a grid by combining multiple batteries-included plots in the same :term:`chart`.

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
    pc_kwargs : mapping, optional
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

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    facet_dims = ["__variable__"] + ([] if "predictive" in group else [dim for dim in distribution.dims if dim not in sample_dims])

    if pc_kwargs is None:
        pc_kwargs = {}
    else:
        pc_kwargs = pc_kwargs.copy()
    pc_kwargs["plot_grid_kws"] = pc_kwargs.get("plot_grid_kws", {}).copy()
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
    pc_kwargs = set_figure_layout(pc_kwargs, plot_bknd, distribution)

    pc = PlotCollection.grid(
        distribution,
        backend=backend,
        **pc_kwargs,
    )

    for name, (plot, kwargs) in zip(plot_names, plots):
        pc_i = plot(
            dt, backend="none", group=group,
            var_names=var_names, filter_vars=filter_vars,
            coords=coords, **kwargs
        )
        pc.coords = None
        pc.aes = pc_i.aes
        pc.coords = {expand: name}
        inverted_dt_dict = {}
        for viz_group, ds in pc_i.viz.children.items():
            for viz_var_name, da in ds.data_vars.items():
                if viz_var_name in {"plot", "row_index", "col_index"}:
                    continue
                if viz_var_name not in inverted_dt_dict:
                    inverted_dt_dict[viz_var_name] = {}
                inverted_dt_dict[viz_var_name].update({viz_group: da})
        inverted_dt_dict = {key: xr.Dataset(values) for key, values in inverted_dt_dict.items()}
        for viz_group, ds in inverted_dt_dict.items():
            attrs = ds[list(ds.data_vars)[0]].attrs
            pc.map(
                render,
                fun_label=f"{viz_group}_{name}",
                data=ds,
                ignore_aes=attrs.get("ignore_aes", None)
            )
    pc.coords = None
    # TODO: at some point all `pc_i.aes` objects should be merged
    # and stored into the `pc.aes` attribute

    return pc
