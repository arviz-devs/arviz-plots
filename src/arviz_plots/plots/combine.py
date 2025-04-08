from arviz_plots import PlotCollection
from arviz_plots.plots.utils import process_group_variables_coords


def combine_plots(
    dt,
    plots=None,
    expanded_dim="column",
    expanded_coord_names=None,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
):
    if expanded_coord_names is None:
        expanded_coord_names = [
            getattr(elem[0], "__name__") + f"_{idx:02d}" for idx, elem in enumerate(plots)
        ]

    column_length = len(expanded_coord_names)

    distribution = process_group_variables_coords(
        dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
    )

    pc_kwargs = {}
    pc_kwargs["plot_grid_kws"] = {}
    if expanded_dim == "column":
        pc_kwargs.setdefault("cols", ["column"])
        pc_kwargs["rows"] = ["__variable__"]
        expand_dims = {"column": column_length}
    if expanded_dim == "row":
        pc_kwargs["cols"] = ["__variable__"]
        pc_kwargs["rows"] = ["row"]
        expand_dims = {"row": column_length}

    pc_kwargs["plot_grid_kws"]["figsize"] = (12, 3)

    pc = PlotCollection.grid(
        distribution.expand_dims(**expand_dims).assign_coords(column=expanded_coord_names),
        backend="matplotlib",
        **pc_kwargs,
    )

    for name, (plot, kwargs) in zip(expanded_coord_names, plots):
        pc.coords = {expanded_dim: name}
        plot(dt, **kwargs, var_names=var_names, group=group, plot_collection=pc)

    return pc
