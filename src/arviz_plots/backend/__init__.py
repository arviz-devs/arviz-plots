"""Common interface to plotting backends.

Each submodule within this module defines a common interface layer to different plotting libraries.

All other modules in ``arviz_subplots`` use this module to interact with the plotting
backends, never interacting directly. Thus, adding a new backend requires only
implementing this common interface for it, with no changes to any of the other modules.
"""
