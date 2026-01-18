# pylint: disable=no-self-use, redefined-outer-name, wrong-import-position
"""Tests specific to the plotly backend."""
import pytest

pytest.importorskip(
    "plotly",
    reason="Plotly not installed, skipping Plotly backend tests",
)

import numpy as np
from plotly.subplots import make_subplots

from arviz_plots.backend.plotly import line
from arviz_plots.backend.plotly.core import PlotlyPlot


@pytest.fixture(scope="function")
def figure():
    return PlotlyPlot(make_subplots(1, 1), 1, 1)


def test_multiple_lines_without_color(figure):
    x = np.arange(4)
    ys = [
        np.sin(x),
        np.cos(x),
    ]

    for y in ys:
        line(x, y, figure)

    assert len(figure.figure.data) == 2
