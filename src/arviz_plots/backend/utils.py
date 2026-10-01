"""Utilities shared by plotting backends."""

import numpy as np
from numpy.typing import ArrayLike


def color_limits(
    values: ArrayLike, vmin: float | None = None, vmax: float | None = None
) -> tuple[float, float]:
    """Return finite color limits, including for constant or nonfinite values."""
    finite_values = np.asarray(values)[np.isfinite(values)]
    low = (
        (float(np.min(finite_values)) if finite_values.size else 0.0)
        if vmin is None
        else float(vmin)
    )
    high = (
        (float(np.max(finite_values)) if finite_values.size else 1.0)
        if vmax is None
        else float(vmax)
    )
    if low == high:
        high = low + 1
    return low, high
