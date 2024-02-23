"""Matplotlib legend generation."""
from matplotlib.lines import Line2D


def legend(target, kwarg_list, label_list, title=None, artist_kwargs=None, **kwargs):
    """Generate a legend on a figure given lists of labels and property kwargs."""
    if artist_kwargs is None:
        artist_kwargs = {}
    kwargs.setdefault("loc", "outside right upper")
    handles = [Line2D([], [], **{**artist_kwargs, **kws}) for kws in kwarg_list]
    return target.legend(handles, label_list, title=title, **kwargs)
