"""Style/templating helpers."""
from pathlib import Path

from arviz_base import rcParams


def use(name):
    """Set an arviz style as the default style/template for all available backends.

    The style will be set for all backends that support it and have it available.
    The supported backends are Matplotlib, Plotly and Bokeh.
    You can use ``arviz_plots.style.available()`` to check which styles are available.
    The ones that works for all backends are listed under the 'common' key.

    Parameters
    ----------
    name : str
        Name of the style to be set as default.

    Raises
    ------
    ValueError
        If the style with the given name is not found.

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
        from bokeh.io import curdoc
        from bokeh.themes import Theme

        path = Path(__file__).parent / "styles" / f"{name}.yml"
        if path.exists():
            curdoc().theme = Theme(filename=str(path))
            ok = True
    except (ImportError, FileNotFoundError):
        pass

    if not ok:
        raise ValueError(f"Style {name} not found.")


def available():
    """List available styles.

    If multiple backends are installed, also lists styles common
    to all backends under the 'common' key.

    Returns
    -------
    dict
        Dictionary with backend names as keys and list of available styles as values.
    """
    styles = {}

    n_backends = 0
    try:
        import matplotlib.pyplot as plt

        styles["matplotlib"] = plt.style.available
        n_backends += 1
    except ImportError:
        pass

    try:
        import plotly.io as pio

        styles["plotly"] = list(pio.templates)
        n_backends += 1
    except ImportError:
        pass

    try:
        from bokeh.themes import built_in_themes

        path = Path(__file__).parent / "styles"
        custom = [file.stem for file in path.glob("*.yml") if path.exists()]
        styles["bokeh"] = list(built_in_themes) + custom
        n_backends += 1
    except ImportError:
        pass

    if n_backends > 1:
        common = set.intersection(*(set(v) for v in styles.values()))
        styles["common"] = list(common)

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
    if backend not in ["matplotlib", "plotly", "bokeh"]:
        raise ValueError(f"Default styles/templates are not supported for Backend {backend}")

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        if name in plt.style.available:
            return plt.style.library[name]

    elif backend == "plotly":
        import plotly.io as pio

        if name in pio.templates:
            return pio.templates[name]

    elif backend == "bokeh":
        from bokeh.themes import Theme

        path = Path(__file__).parent / "styles" / f"{name}.yml"
        if path.exists():
            return Theme(filename=path)

    raise ValueError(f"Style {name} not found.")
