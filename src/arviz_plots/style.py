"""Style/templating helpers."""
import os

from arviz_base import rcParams


def use(name):
    """Set an arviz style as the default style/template for all available backends.

    Parameters
    ----------
    name : str
        Name of the style to be set as default.
    """
    ok = False

    try:
        import matplotlib.pyplot as plt

        if name in plt.style.available:
            plt.style.use(name)
            ok = True
    except ImportError:
        pass

    try:
        import plotly.io as pio

        if name in pio.templates:
            pio.templates.default = name
            ok = True
    except ImportError:
        pass

    try:
        if name in ["arviz-cetrino", "arviz-tenui", "arviz-variat", "arviz-vibrant"]:
            from bokeh.io import curdoc
            from bokeh.themes import Theme

            path = os.path.dirname(os.path.abspath(__file__))
            curdoc().theme = Theme(filename=f"{path}/styles/{name}.yml")
            ok = True
    except (ImportError, FileNotFoundError):
        pass

    if not ok:
        raise ValueError(f"Style {name} not found.")


def available():
    """List available styles."""
    styles = {}

    try:
        import matplotlib.pyplot as plt

        styles["matplotlib"] = plt.style.available
    except ImportError:
        pass

    try:
        import plotly.io as pio

        styles["plotly"] = list(pio.templates)
    except ImportError:
        pass

    return styles


def get(name, backend=None):
    """Get the style/template with the given name.

    Parameters
    ----------
    name : str
        Name of the style/template to get.
    backend : str
        Name of the backend to use. Options are 'matplotlib' and 'plotly'.
        Defaults to ``rcParams["plot.backend"]``.
    """
    if backend is None:
        backend = rcParams["plot.backend"]
    if backend not in ["matplotlib", "plotly"]:
        raise ValueError(f"Default styles/templates are not supported for Backend {backend}")

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        if name in plt.style.available:
            return plt.style.library[name]

    elif backend == "plotly":
        import plotly.io as pio

        if name in pio.templates:
            return pio.templates[name]

    raise ValueError(f"Style {name} not found.")
