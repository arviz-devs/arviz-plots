"""None backend manual legend generation.

For now only used for documentation purposes.
"""


# pylint: disable=unused-argument
def legend(
    plot_collection,
    kwarg_list,
    label_list,
    title=None,
    visual_type="line",
    visual_kwargs=None,
    legend_dim=None,
    update_visuals=None,
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
        When defining a miniature, both the _i_-th element in `kwarg_list` and
        `visual_kwargs` are passed to the backend. If a key is in both places,
        the one in `kwarg_list` should take priority.
    legend_dim : str or sequence of str, optional
        Dimension or dimensions whose mappings should be used to generate the legend.
    update_visuals : bool, optional
        If relevant for the backend, update objects representing :term:`visual` elements
        of the plot to improve or allow interactivity for the legend.
    **kwargs
        Passed to backend legend generating function.

    Returns
    -------
    legend : legend object
        A scalar backend object representing the legend
    """
    return None
