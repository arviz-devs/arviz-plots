"""Batteries-included ArviZ plots."""

from .posteriorplot import plot_posterior
from .traceplot import plot_trace

__all__ = ["plot_posterior", "plot_trace"]
