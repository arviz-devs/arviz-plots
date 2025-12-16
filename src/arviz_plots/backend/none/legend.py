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
    label_list : sequence of str
    title : str, optional
    visual_type : {"line", "scatter", "rectangle"}, default "line"
    visual_kwargs : mapping, optional
        Passed to all visuals when generating legend miniatures.
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
