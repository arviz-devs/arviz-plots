"""Batteries-included ArviZ plots."""

from .posteriorplot import plot_dist
from .traceplot import plot_trace

__all__ = ["plot_dist", "plot_trace"]
