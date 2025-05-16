"""None backend manual legend generation.

For now only used for documentation purposes.
"""


# pylint: disable=unused-argument
def legend(
    target, kwarg_list, label_list, title=None, artist_type="line", artist_kwargs=None, **kwargs
):
    """Generate a legend on a figure given lists of labels and property kwargs.

    Parameters
    ----------
    target : plot object
    kwarg_list : sequence of mapping
    label_list : sequence of str
    title : str, optional
    artist_type : {"line", "scatter", "rectangle"}, default "line"
    artist_kwargs : mapping, optional
        Passed to all artists when generating legend miniatures.
    **kwargs
        Passed to backend legend generating function.
    """
    return None
