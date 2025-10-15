"""Backend agnostic utilities for handling arguments and aliases."""
import re


def get_contrast_colors(bg_color="#FFFFFF"):
    """Get contrast colors based on the background color.

    Checks the brightness of the provided background color and provides colors
    that will contrast with white if brightness is high, and colors
    that contrast with black otherwise.

    It will not work for arbitrary colorful backgrounds not it is within its scope.
    """
    color = bg_color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    # calculating the YIQ brightness value
    yiq = (r * 299 + g * 587 + b * 114) / 1000
    return ("#FFFFFF", "#959595", "#5B5B5B") if yiq < 128 else ("#000000", "#595959", "#949494")


def create_aesthetic_handlers(get_default_aes, get_background_color):
    """Create aesthetic alias handling functions for a backend.

    The main aliases are things like "C0", "C13", "C5"... which indicate to use the n-th
    element of the default cycler for that aesthetic. This is available
    for color, facecolor, edgecolor, marker and linestyle.

    Moreover, the first three properties which are color based also allow
    aliases for colors that depend on the background color:

    * B0 -> the background color of the currently active theme
    * B1 -> a color with high contrast with B0 (21:1 contrast for black or white backgrounds)
    * B2 -> a color with muted contrast with B0 (7:1 contrast for black or white backgrounds).
      This color is more muted than B1 but still high contrast enough for accessible text
      within image or thin visual elements.
    * B3 -> a color with muted contrast with B0 (3:1 contrast for black or white background).
      This color is more muted than B2 and might not be easily readable for everyone.
      It is still adequate for complementary graphical elements like grid lines or borders.

    Parameters
    ----------
    get_default_aes : callable
        Backend-specific function that generates default aesthetic values.
        Should have signature: get_default_aes(aes_key, n, kwargs=None)
    get_background_color : callable
        Backend-specific function that returns the background color of
        the currently active theme. It is called without arguments.
    """
    cycle_pattern = re.compile(r"C[0-9]+")
    background_pattern = re.compile(r"B[0-3]")

    def dealiase_aes_value(aes, value):
        """Convert aesthetic aliases like 'C0', 'B1' to actual values."""
        index = int(value[1:])
        if value.startswith("C"):
            dealiased_value = get_default_aes(aes, index + 1)[index]
        else:
            bg_color = get_background_color()
            match index:
                case 0:
                    dealiased_value = bg_color
                case 1:
                    dealiased_value = get_contrast_colors(bg_color=bg_color)[0]
                case 2:
                    dealiased_value = get_contrast_colors(bg_color=bg_color)[1]
                case 3:
                    dealiased_value = get_contrast_colors(bg_color=bg_color)[2]
                case _:
                    raise ValueError("Unrecognized background dependent color alias")
        return dealiased_value

    def expand_aesthetic_aliases(plot_fn):
        def _dealiased_plot_fn(*args, **kwargs):
            for aes in ("color", "facecolor", "edgecolor"):
                if (
                    aes in kwargs
                    and isinstance(value := kwargs[aes], str)
                    and (cycle_pattern.match(value) or background_pattern.match(value))
                ):
                    kwargs[aes] = dealiase_aes_value(aes, value)
            for aes in ("linestyle", "marker"):
                if (
                    aes in kwargs
                    and isinstance(value := kwargs[aes], str)
                    and cycle_pattern.match(value)
                ):
                    kwargs[aes] = dealiase_aes_value(aes, value)
            return plot_fn(*args, **kwargs)

        return _dealiased_plot_fn

    return expand_aesthetic_aliases
