"""Bokeh manual legend generation."""
import warnings

import numpy as np
from bokeh.models import Legend


def dealiase_line_kwargs(kwargs):
    """Convert arviz common interface properties to bokeh ones."""
    prop_map = {"width": "line_width", "linestyle": "line_dash"}
    return {prop_map.get(key, key): value for key, value in kwargs.items()}


def legend(
    target,
    kwarg_list,
    label_list,
    title=None,
    artist_type="line",
    artist_kwargs=None,
    legend_target=None,
    side="right",
    **kwargs,
):
    """Generate a legend on a figure given lists of labels and property kwargs.

    Parameters
    ----------
    legend_target : (int, int), default (0, -1)
        Row and colum indicators of the :term:`plot` where the legend will be placed.
        Bokeh does not support :term:`chart` level legend.
    side : str, optional
        Side of the plot on which to place the legend. Use "center" to put the legend
        inside the plotting area.
    """
    if artist_kwargs is None:
        artist_kwargs = {}
    if legend_target is None:
        legend_target = (0, -1)
    # TODO: improve selection of Figure object from what is stored as "chart"
    children = target.children
    if not isinstance(children[0], tuple):
        children = children[1].children
    plots = [child[0] for child in children]
    row_id = np.array([child[1] for child in children], dtype=int)
    col_id = np.array([child[2] for child in children], dtype=int)
    legend_id = np.argmax(
        (row_id == np.unique(row_id)[legend_target[0]])
        & (col_id == np.unique(col_id)[legend_target[1]])
    )
    target_plot = plots[legend_id]
    if target_plot.legend:
        warnings.warn("This target plot already contains a legend")
    glyph_list = []
    if artist_type == "line":
        artist_fun = target_plot.line
        kwarg_list = [dealiase_line_kwargs(kws) for kws in kwarg_list]
    else:
        raise NotImplementedError("Only line type legends supported for now")

    for kws in kwarg_list:
        glyph = artist_fun(**{**artist_kwargs, **kws})
        glyph_list.append(glyph)
    leg = Legend(
        items=[(str(label), [glyph]) for label, glyph in zip(label_list, glyph_list)],
        title=title,
        **kwargs,
    )
    target_plot.add_layout(leg, side)
    return leg
