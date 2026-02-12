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

from arviz_plots.backend.plotly import fill_between_y, line, multiple_lines
from arviz_plots.backend.plotly.core import PlotlyPlot

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


def test_multiple_lines(figure):
    x = np.array([0, 1, 2, 3])
    ys = np.array([[10, 100], [11, 101], [12, 102], [13, 103]])
    n_lines = ys.shape[1]

    lines_obj = multiple_lines(x, ys, figure)
    assert len(figure.figure.data) == 1

    for visual in (lines_obj, figure.data[0]):
        assert isinstance(visual, go.Scatter)
        assert visual.mode == "lines"
        assert visual.showlegend is False
        stitches = visual.x[~np.isin(visual.x, x)]
        assert np.isnan(stitches).all()
        assert visual.y.size == (ys.size + n_lines)
        assert np.isnan(visual.y).sum() == n_lines


def test_multiple_lines_args(figure):
    x = np.array([0, 1, 2, 3])
    ys = np.array([[10, 100], [11, 101], [12, 102], [13, 103]])

    lines_obj = multiple_lines(x, ys, figure, color="red", alpha=0.5, width=2)
    assert len(figure.figure.data) == 1

    for visual in (lines_obj, figure.data[0]):
        assert "color" in visual.line
        assert visual.line["color"] == "rgba(255, 0, 0, 0.500)"
        assert "width" in visual.line
        assert visual.line["width"] == 2


def test_fill_between_y(figure):
    x = np.array([0, 1, 2, 3])
    y1 = np.sin(x)
    y2 = np.cos(x)
    fill_line_obj = fill_between_y(x, y1, y2, figure)
    assert len(figure.figure.data) == 2

    trace1, trace2 = figure.figure.data
    assert trace1.type == "scatter"
    assert trace1.mode == "lines"
    assert trace1.fill is None
    assert trace1.showlegend is False
    assert np.array_equal(trace1.x, x)
    assert np.array_equal(trace1.y, y1)

    for visual in (fill_line_obj, trace2):
        assert visual.type == "scatter"
        assert visual.mode == "none"
        assert visual.fill == "tonexty"
        assert visual.showlegend is False
        assert np.array_equal(visual.x, x)
        assert np.array_equal(visual.y, y2)


def test_fill_between_y_args(figure):
    x = np.array([0, 1, 2, 3])
    y1 = np.sin(x)
    y2 = np.cos(x)
    fill_line_obj = fill_between_y(x, y1, y2, figure, color="red", alpha=0.5)
    assert len(figure.figure.data) == 2

    for visual in (fill_line_obj, figure.figure.data[1]):
        assert visual["fillcolor"] == "rgba(255, 0, 0, 0.500)"
