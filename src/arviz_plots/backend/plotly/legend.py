"""Plotly legend generation."""


def dealiase_line_kwargs(kwargs):
    """Convert arviz common interface properties to plotly ones."""
    prop_map = {"linewidth": "width", "linestyle": "dash"}
    return {prop_map.get(key, key): value for key, value in kwargs.items()}


def legend(
    target, kwarg_list, label_list, title=None, artist_type="line", artist_kwargs=None, **kwargs
):
    """Generate a legend with plotly.

    Parameters
    ----------
    target : plotly.graph_objects.Figure
        The figure to add the legend to
    kwarg_list : list
        List of style dictionaries for each legend entry
    label_list : list
        List of labels for each legend entry
    title : str, optional
        Title of the legend
    artist_type : str, optional
        Type of artist to use for legend entries. Currently only "line" is supported.
    artist_kwargs : dict, optional
        Additional kwargs passed to all artists
    **kwargs : dict
        Additional kwargs passed to legend configuration

    Returns
    -------
    None
        The legend is added to the target figure inplace
    """
    if artist_kwargs is None:
        artist_kwargs = {}

    if artist_type == "line":
        artist_fun = target.add_scatter
        kwarg_list = [dealiase_line_kwargs(kws) for kws in kwarg_list]
        mode = "lines"
    else:
        raise NotImplementedError("Only line type legends supported for now")

    for kws, label in zip(kwarg_list, label_list):
        artist_fun(
            x=[None],
            y=[None],
            name=str(label),
            mode=mode,
            line=kws,
            showlegend=True,
            **artist_kwargs,
        )

    target.update_layout(showlegend=True, legend_title_text=title, **kwargs)
