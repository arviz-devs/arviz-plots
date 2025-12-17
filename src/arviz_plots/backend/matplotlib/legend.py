"""Matplotlib manual legend generation."""
from matplotlib.lines import Line2D

from .core import expand_aesthetic_aliases


@expand_aesthetic_aliases
def dealiase_line_kwargs(**kwargs):
    """Convert arviz common interface properties to matplotlib ones."""
    prop_map = {"width": "linewidth"}
    return {prop_map.get(key, key): value for key, value in kwargs.items()}


def legend(
    plot_collection,
    kwarg_list,
    label_list,
    title=None,
    visual_type="line",
    visual_kwargs=None,
    legend_dim=None,  # pylint: disable=unused-argument
    update_visuals=True,  # pylint: disable=unused-argument
    **kwargs,
):
    """Generate a legend on a figure given lists of labels and property kwargs."""
    if visual_kwargs is None:
        visual_kwargs = {}
    if "legend" in plot_collection.viz.children:
        legend_number = len(plot_collection.viz["legend"].data_vars) + 1
    else:
        legend_number = 1
    default_y = {1: "right upper", 2: "right lower", 3: "center right"}[legend_number]
    kwargs.setdefault("loc", f"outside {default_y}")
    if visual_type == "line":
        visual_fun = Line2D
        kwarg_list = [dealiase_line_kwargs(**kws) for kws in kwarg_list]
    else:
        raise NotImplementedError("Only line type legends supported for now")
    handles = [visual_fun([], [], **{**visual_kwargs, **kws}) for kws in kwarg_list]
    figure = plot_collection.get_viz("figure")
    return figure.legend(handles, label_list, title=title, **kwargs)
