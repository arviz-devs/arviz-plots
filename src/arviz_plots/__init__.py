# pylint: disable=wildcard-import,wrong-import-position
"""ArviZ plots."""

import logging

_log = logging.getLogger(__name__)

from ._version import __version__

from .plot_collection import PlotMuseum
from .plots import *
