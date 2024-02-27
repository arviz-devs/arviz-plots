"""Matplotlib manual legend generation."""
from matplotlib.lines import Line2D

def dealiase_line_kwargs(kwargs):
    """Convert arviz common interface properties to matplotlib ones."""
    prop_map = {"width": "linewidth"}
    return {prop_map.get(key, key): value for key, value in kwargs.items()}


def legend(target, kwarg_list, label_list, title=None, artist_type="line", artist_kwargs=None, **kwargs):
    """Generate a legend on a figure given lists of labels and property kwargs."""
    if artist_kwargs is None:
        artist_kwargs = {}
    kwargs.setdefault("loc", "outside right upper")
    if artist_type == "line":
        artist_fun = Line2D
        kwarg_list = [dealiase_line_kwargs(kws) for kws in kwarg_list]
    else:
        raise NotImplementedError("Only line type legends supported for now")
    handles = [artist_fun([], [], **{**artist_kwargs, **kws}) for kws in kwarg_list]
    return target.legend(handles, label_list, title=title, **kwargs)
