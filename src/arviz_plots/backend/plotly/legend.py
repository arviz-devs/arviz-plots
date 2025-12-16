"""Plotly legend generation."""
import numpy as np
import xarray as xr
from arviz_base import xarray_sel_iter
from plotly.graph_objects import Bar, Contour, Heatmap, Scatter

from .core import expand_aesthetic_aliases


@expand_aesthetic_aliases
def dealiase_line_kwargs(**kwargs):
    """Convert arviz common interface properties to plotly ones."""
    prop_map = {"linewidth": "width", "linestyle": "dash"}
    return {prop_map.get(key, key): value for key, value in kwargs.items()}


LINE_SUBKEYS = [
    "backoff",
    "backoffsrc",
    "color",
    "dash",
    "shape",
    "simplify",
    "smoothing",
    "width",
]


def legend(
    plot_collection,
    kwarg_list,
    label_list,
    title=None,
    visual_type="line",
    visual_kwargs=None,
    legend_dim=None,
    update_visuals=True,
    **kwargs,
):
    """Generate a legend with plotly.

    Parameters
    ----------
    plot_collection : PlotCollection
        The PlotCollection for which a legend should be generated
    kwarg_list : list
        List of style dictionaries for each legend entry
    label_list : list
        List of labels for each legend entry
    title : str, optional
        Title of the legend
    artist_type : str, optional
        Type of visual to use for legend entries. Currently only "line" is supported.
    artist_kwargs : dict, optional
        Additional kwargs passed to all visuals
    legend_dim : str or tuple of str, optional
    update_visuals : bool, default True
    **kwargs : dict
        Additional kwargs passed to legend configuration

    Returns
    -------
    None
        The legend is added to the target figure inplace
    """
    figure = plot_collection.get_viz("figure")
    if "legend" in plot_collection.viz.children:
        legend_number = len(plot_collection.viz["legend"].data_vars) + 1
        legend_id = f"legend{legend_number}"
    else:
        legend_number = 1
        legend_id = "legend"
    kwargs.setdefault("legend_y", {1: 1, 2: 0, 3: 0.5}[legend_number])
    kwargs["legend_title_text"] = title
    legend_kwargs = kwargs.pop(legend_id, {}).copy()
    kwargs_list = list(kwargs.items())
    for key, value in kwargs_list:
        if key.startswith("legend"):
            kwargs.pop(key)
            legend_kwargs[key[len("legend_") :]] = value
    kwargs[legend_id] = legend_kwargs
    if visual_kwargs is None:
        visual_kwargs = {}
    else:
        visual_kwargs = visual_kwargs.copy()

    if visual_type == "line":
        visual_fun = figure.add_scatter
        kwarg_list = [dealiase_line_kwargs(**kws) for kws in kwarg_list]
        mode = "lines"
        visual_kwargs.setdefault("line_color", "black")
    else:
        raise NotImplementedError("Only line type legends supported for now")

    if update_visuals:
        for group, viz_data in plot_collection.viz.children.items():
            if group in {"plot", "row_index", "col_index"}:
                continue
            viz_ds = viz_data.dataset
            if any((d not in viz_ds.dims) and (d != "__variable__") for d in legend_dim):
                continue
            for var_name, sel, _ in xarray_sel_iter(viz_ds, skip_dims={}):
                target_viz = viz_ds[var_name].sel(sel).item()
                if target_viz is None:
                    continue
                if not isinstance(target_viz, (Scatter, Bar)):
                    break

                def trace_matcher(trace):
                    return (
                        (getattr(trace, "mode", "na") == getattr(target_viz, "mode", "na"))
                        and (getattr(trace, "line", "na") == getattr(target_viz, "line", "na"))
                        and (trace.marker == target_viz.marker)
                        and (trace.text == target_viz.text)
                        and (trace.x.shape == target_viz.x.shape)
                        and np.allclose(trace.y, target_viz.y)
                        and np.allclose(trace.x, target_viz.x)
                    )

                target_plot = plot_collection.get_target(var_name, sel)
                if isinstance(target_plot, xr.DataArray):
                    target_plot = target_plot.data
                else:
                    target_plot = [target_plot]
                for element in target_plot:
                    element.update_traces(
                        selector=trace_matcher,
                        **plot_collection.get_aes_kwargs(["legendgroup"], var_name, sel),
                    )

    for kws, label in zip(kwarg_list, label_list):
        # plotly allow passing arguments as `line={key: value}` or as `line_key=value` directly
        # the 2nd option allows for more user flexibility in overriding or extending kwargs
        kws = {f"line_{key}" if key in LINE_SUBKEYS else key: value for key, value in kws.items()}
        visual_fun(
            x=[None],
            y=[None],
            name=str(label),
            mode=mode,
            showlegend=True,
            legend=legend_id,
            **{**visual_kwargs, **kws},
        )

    figure.update_layout(showlegend=True, **kwargs)
    return figure.layout.legend
