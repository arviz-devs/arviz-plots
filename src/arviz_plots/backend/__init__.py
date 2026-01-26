"""Common interface to plotting backends.

Each submodule within this module defines a common interface layer to different plotting libraries.

Outside ``arviz_plots.backend`` the corresponding backend module is imported,
but only the common interface layer is used, making no distinctions between plotting backends.
Each submodule inside ``arviz_plots.backend`` is expected to implement the same functions
with the same call signature. Thus, adding a new backend requires only
implementing this common interface for it, with no changes to any of the other modules.
"""
from arviz_plots.backend.none import *
