import re
from importlib import import_module

import xarray as xr
from arviz_base import rcParams

from arviz_plots import PlotCollection
from arviz_plots.plots.utils import process_group_variables_coords

def get_valid_arg(key, value, backend):
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    key_matcher = "color" if key in {"facecolor", "edgecolor"} else key
    if isinstance(value, str) and re.match(key_matcher +"_[0-9]+", value):
        _, index = value.rsplit("_", 1)
        index = int(index)
        return plot_backend.get_default_aes(key, index+1)[index]
    return value


def backendize_kwargs(kwargs, backend):
    return {key: get_valid_arg(key, value, backend) for key, value in kwargs.items() if key != "function"}

def render(da, target, backend, **kwargs):
    plot_backend = import_module(f"arviz_plots.backend.{backend}")
    plot_kwargs = da.item()
    plot_fun_name = plot_kwargs["function"]
    plot_kwargs = backendize_kwargs(plot_kwargs, backend)
    return getattr(plot_backend, plot_fun_name)(target=target, **{**plot_kwargs, **kwargs})

def combine_plots(
    dt,
    plots=None,
    expanded_dim="column",
    expanded_coord_names=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    backend="matplotlib",
):
    if expanded_coord_names is None:
        expanded_coord_names = [
            getattr(elem[0], "__name__") + f"_{idx:02d}" for idx, elem in enumerate(plots)
        ]
    sample_dims = rcParams["data.sample_dims"]
    column_length = len(expanded_coord_names)

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )
    facet_dims = ["__variable__"] + ([] if "predictive" in group else [dim for dim in distribution.dims if dim not in sample_dims])

    pc_kwargs = {}
    pc_kwargs["plot_grid_kws"] = {}
    if expanded_dim == "column":
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs["rows"] = facet_dims
        expand_dims = {"column": column_length}
    elif expanded_dim == "row":
        pc_kwargs["cols"] = facet_dims
        pc_kwargs["rows"] = ["row"]
        expand_dims = {"row": column_length}
    else:
        raise ValueError()

    pc_kwargs["plot_grid_kws"]["figsize"] = (12, 3)

    pc = PlotCollection.grid(
        distribution.expand_dims(**expand_dims).assign_coords({expanded_dim: expanded_coord_names}),
        backend=backend,
        **pc_kwargs,
    )

    for name, (plot, kwargs) in zip(expanded_coord_names, plots):
        pc_i = plot(
            dt, backend="none", group=group,
            var_names=var_names, filter_vars=filter_vars,
            coords=coords, **kwargs
        )
        pc.coords = None
        pc.aes = pc_i.aes
        pc.coords = {expanded_dim: name}
        inverted_dt = {}
        for viz_group, ds in pc_i.viz.children.items():
            for viz_var_name, da in ds.data_vars.items():
                if viz_var_name in {"plot", "row_index", "col_index"}:
                    continue
                if viz_var_name not in inverted_dt:
                    inverted_dt[viz_var_name] = {}
                inverted_dt[viz_var_name].update({viz_group: da})
        inverted_dt = xr.DataTree.from_dict({key: xr.Dataset(values) for key, values in inverted_dt.items()})
        for viz_group, ds in inverted_dt.children.items():
            pc.map(render, data=ds.dataset, store_artist=False)

    return pc
