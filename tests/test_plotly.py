# pylint: disable=no-self-use, redefined-outer-name, wrong-import-position
"""Tests specific to the plotly backend."""
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from arviz_plots.backend.plotly import line
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
