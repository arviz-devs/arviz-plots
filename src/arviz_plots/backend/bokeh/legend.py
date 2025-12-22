"""Bokeh manual legend generation."""
import warnings

from bokeh.models import Legend

from .core import expand_aesthetic_aliases


@expand_aesthetic_aliases
def dealiase_line_kwargs(**kwargs):
    """Convert arviz common interface properties to bokeh ones."""
    prop_map = {"width": "line_width", "linestyle": "line_dash"}
    return {prop_map.get(key, key): value for key, value in kwargs.items()}


def legend(
    plot_collection,
    kwarg_list,
    label_list,
    title=None,
    visual_type="line",
    visual_kwargs=None,
    legend_dim=None,  # pylint: disable=unused-argument
    update_visuals=False,  # pylint: disable=unused-argument
    legend_target=None,
    side="right",
    **kwargs,
):
    """Generate a legend on a figure given lists of labels and property kwargs.

    Parameters
    ----------
    plot_collection : PlotCollection
    kwarg_list : sequence of mapping
        Sequence with length equal to the number of entries to add to the legend.
        The elements in the list are the kwargs to use when defining the legend
        miniatures.
    label_list : sequence of str
        Sequence with length equal to the number of entries to add to the legend.
        The elements in the list are the labels to give each miniature in the legend.
    title : str, optional
        The title to give the legend.
    visual_type : {"line", "scatter", "rectangle"}, default "line"
    visual_kwargs : mapping, optional
        Passed to all visuals when generating legend miniatures.
        For "line" visual type passed to :meth:`bokeh.plotting.figure.line`
    legend_dim : str or sequence of str, optional
        Dimension or dimensions whose mappings should be used to generate the legend.
    update_visuals : bool, optional
        If relevant for the backend, update objects representing :term:`visual` elements
        of the plot to improve or allow interactivity for the legend.
    legend_target : (int, int), default (0, -1)
        Row and colum indicators of the :term:`plot` where the legend will be placed.
        Bokeh does not support :term:`figure` level legend.
    side : str, optional
        Side of the plot on which to place the legend. Use "center" to put the legend
        inside the plotting area.
    **kwargs
        Passed to :class:`bokeh.models.Legend`
    """
    if visual_kwargs is None:
        visual_kwargs = {}
    if legend_target is None:
        legend_target = (0, -1)
    if side == "right":
        kwargs.setdefault("margin", -55)

    target_plot = plot_collection.iget_target(*legend_target)
    if target_plot.legend:
        warnings.warn("This target plot already contains a legend")
    glyph_list = []
    if visual_type == "line":
        visual_fun = target_plot.line
        kwarg_list = [dealiase_line_kwargs(**kws) for kws in kwarg_list]
    else:
        raise NotImplementedError("Only line type legends supported for now")

    for kws in kwarg_list:
        glyph = visual_fun(**{**visual_kwargs, **kws})
        glyph_list.append(glyph)

    leg = Legend(
        items=[(str(label), [glyph]) for label, glyph in zip(label_list, glyph_list)],
        title=title,
        **kwargs,
    )

    target_plot.add_layout(leg, side)
    return leg
