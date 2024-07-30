"""Style/templating helpers."""


def use(name):
    """Set an arviz style as the default style/template for all available backends.

    Parameters
    ----------
    name : str
        Name of the style to be set as default.

    Notes
    -----
    There are some backends where default styles/templates are not supported.
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

    if not ok:
        raise ValueError(f"Style {name} not found.")


def available():
    """List available styles."""
    try:
        import matplotlib.pyplot as plt

        print("Matplotlib styles:")
        print(plt.style.available)
        print()
    except ImportError:
        pass

    try:
        import plotly.io as pio

        print("Plotly templates:")
        print(list(pio.templates))
    except ImportError:
        pass


def get(name, backend="matplotlib"):
    """Get the style/template with the given name.

    Parameters
    ----------
    name : str
        Name of the style/template to get.
    backend : str
        Name of the backend to use. Options are 'matplotlib' (default) and 'plotly'.
    """
    if backend not in ["matplotlib", "plotly"]:
        raise ValueError(f"Default styles/templates are not supported for Backend {backend}")

    if backend == "matplotlib":
        try:
            import matplotlib.pyplot as plt

            if name in plt.style.available:
                return plt.style.library[name]
        except ImportError:
            pass

    elif backend == "plotly":
        try:
            import plotly.io as pio

            if name in pio.templates:
                return pio.templates[name]
        except ImportError:
            pass

    return ValueError(f"Style {name} not found.")
