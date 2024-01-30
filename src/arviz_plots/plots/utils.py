"""Utilities for batteries included plots."""


def filter_aes(pc, aes_map, artist, sample_dims):
    """Split aesthetics and get relevant dimensions.

    Returns
    -------
    artist_dims : list
        Dimensions that should be reduced for this artist.
        That is, all dimensions in `sample_dims` that are not
        mapped to any aesthetic.
    artist_aes : iterable
    ignore_aes : set
    """
    artist_aes = aes_map.get(artist, {})
    pc_aes = pc.aes_set
    ignore_aes = set(pc_aes).difference(artist_aes)
    _, all_loop_dims = pc.update_aes(ignore_aes=ignore_aes)
    artist_dims = [dim for dim in sample_dims if dim not in all_loop_dims]
    return artist_dims, artist_aes, ignore_aes


def get_size_of_var(group, compact=False):
    """Get the size of the variables in a group."""
    coords = set(group.sizes) - set(sample_dims)
    var_size = 0
    for var in group.data_vars:
        dims_size = group[var].sizes
        partial_sum = 1
        if not compact:
            for key in coords:
                if key in dims_size:
                    partial_sum *= dims_size[key]

        var_size += partial_sum

    return var_size


def scale_fig_size(figsize, rows=1, cols=1):
    """Scale figure properties according to figsize, rows and cols.

    Parameters
    ----------
    figsize : float or None
        Size of figure in inches
    textsize : float or None
        fontsize
    rows : int
        Number of rows
    cols : int
        Number of columns

    Returns
    -------
    figsize : float or None
        Size of figure in inches
    labelsize : int
        fontsize for labels
    linewidth : int
        linewidth
    """
    # we should read figsize from rcParams or bokeh theme
    if figsize is None:
        width = 8
        height = (rows + 1) ** 1.1
    else:
        width, height = figsize

    # we should read textsize from rcParams or bokeh theme
    textsize = 14
    val = (width * height) ** 0.5
    val2 = (cols * rows) ** 0.5
    labelsize = textsize * (val / 4) / val2
    linewidth = labelsize / 10

    return (width, height), labelsize, linewidth
