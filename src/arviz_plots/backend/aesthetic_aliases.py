"""Common aesthetic alias handling functionality for backends."""


def create_aesthetic_handlers(get_default_aes):
    """Create aesthetic alias handling functions for a backend.

    Parameters
    ----------
    get_default_aes : callable
        Backend-specific function that generates default aesthetic values.
        Should have signature: get_default_aes(aes_key, n, kwargs=None)
    """

    def dealiase_aes_value(aes, value):
        """Convert aesthetic aliases like 'C0', 'color_1' to actual values."""
        try:
            if value.startswith(f"{aes}_"):
                index = int(value.rsplit("_")[-1])
            else:
                index = int(value[1:])
            dealiased_value = get_default_aes(aes, index + 1)[index]
        except ValueError:
            return value
        return dealiased_value

    def expand_aesthetic_aliases(plot_fn):
        def _dealiased_plot_fn(*args, **kwargs):
            for aes in ("color", "facecolor", "edgecolor"):
                if (
                    aes in kwargs
                    and isinstance(value := kwargs[aes], str)
                    and value[0] in ("C", aes[0])
                    and (value[1].isdigit())
                ):
                    kwargs[aes] = dealiase_aes_value(aes, value)
            for aes in ("linestyle", "marker"):
                if (
                    aes in kwargs
                    and isinstance(value := kwargs[aes], str)
                    and value[0] in ("C", aes[0])
                    and (value[1].isdigit())
                ):
                    kwargs[aes] = dealiase_aes_value(aes, value)
            return plot_fn(*args, **kwargs)

        return _dealiased_plot_fn

    return expand_aesthetic_aliases
