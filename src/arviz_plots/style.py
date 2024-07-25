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
    try:
        import matplotlib.pyplot as plt

        plt.style.use(name)
    except ImportError:
        pass

    try:
        import plotly.io as pio

        pio.templates.default = name
    except ImportError:
        pass
