# File generated with docstub

import logging
import os

from _typeshed import Incomplete

_log: Incomplete

from arviz_plots import style, visuals
from arviz_plots._version import __version__
from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plot_matrix import PlotMatrix
from arviz_plots.plots import *

if not logging.root.handlers:
    _handler: Incomplete
    _formatter: Incomplete

try:
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.pyplot import style as mplstyle

    _arviz_style_path: Incomplete
    if hasattr(mplstyle, "USER_LIBRARY_PATHS"):
        pass
    else:
        pass

    _linear_grey_10_95_c0: Incomplete

    def _mpl_cm(name: Incomplete, colorlist: Incomplete) -> None: ...

    try:
        import colorcet
    except ModuleNotFoundError:
        pass

    del LinearSegmentedColormap, mpl, mplstyle

except ImportError:
    pass

try:
    import plotly.io as pio

    from arviz_plots.backend.plotly.templates import (
        arviz_cetrino_template,
        arviz_darkgrid_template,
        arviz_tenui_template,
        arviz_tumma_template,
        arviz_variat_template,
        arviz_vibrant_template,
    )

    pio: Incomplete
    templates: Incomplete
    pio: Incomplete
    templates: Incomplete
    pio: Incomplete
    templates: Incomplete
    pio: Incomplete
    templates: Incomplete
    pio: Incomplete
    templates: Incomplete
    pio: Incomplete
    templates: Incomplete

except ImportError:
    pass

del os, logging
