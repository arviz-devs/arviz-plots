# pylint: disable=no-self-use, redefined-outer-name, wrong-import-position
"""Tests specific to the plotly backend."""
import os
import numpy as np
import pytest

if os.environ.get("ARVIZ_REQUIRE_ALL_DEPS", False):
    import plotly  # pylint: disable=unused-import
else:
    pytest.importorskip("plotly")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from arviz_plots.backend.plotly import line
from arviz_plots.backend.plotly.core import PlotlyPlot, fill_between_y, multiple_lines

pytestmark = [pytest.mark.usefixtures("check_skips"), pytest.mark.plotly]


@pytest.fixture(scope="function")
def figure():
    return PlotlyPlot(make_subplots(1, 1), 1, 1)


def test_line(figure):
    line_obj = line([0, 1, 2], [0, 2, 1], figure)
    assert len(figure.data) == 1
    # we can't test with `is` nor with equality between `line_obj` and `figure.data[0]`
    for visual in (line_obj, figure.data[0]):
        assert isinstance(visual, go.Scatter)
        assert visual.mode == "lines"
        assert visual.showlegend is False


def test_line_args(figure):
    line_obj = line([0, 1, 2], [0, 2, 1], figure, color="orange", width=2.2)
    for visual in (line_obj, figure.data[0]):
        assert "color" in visual.line
        assert visual.line["color"] == "orange"
        assert "width" in visual.line
        assert visual.line["width"] == 2.2

        
def test_multiple_lines_without_color(figure):
    x = np.array([0, 1, 2, 3])
    ys = np.array(
        [
            [10, 100],
            [11, 101],
            [12, 102],
            [13, 103],
        ]
    )

    multiple_lines(x, ys, figure)

    assert len(figure.figure.data) == 1

    trace = figure.figure.data[0]

    assert trace.mode == "lines"

    assert np.isnan(trace.x).any()

    assert np.isnan(trace.y).any()


def test_fill_between_y_without_color(figure):
    x = np.array([0, 1, 2, 3])
    y1 = np.sin(x)
    y2 = np.cos(x)
    fill_between_y(x, y1, y2, figure)

    assert len(figure.figure.data) == 2

    trace1, trace2 = figure.figure.data

    assert trace1.type == "scatter"
    assert trace2.type == "scatter"

    assert trace1.mode == "lines"
    assert trace2.mode == "none"

    assert trace1.fill is None
    assert trace2.fill == "tonexty"

    assert np.array_equal(trace1.x, x)
    assert np.array_equal(trace2.x, x)

