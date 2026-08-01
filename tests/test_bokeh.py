# pylint: disable=redefined-outer-name, wrong-import-position
"""Tests specific to the bokeh backend."""

import os

import numpy as np
import pytest

if os.environ.get("ARVIZ_REQUIRE_ALL_DEPS", False):
    import bokeh  # noqa: F401  # pylint: disable=unused-import
else:
    pytest.importorskip("bokeh")

from bokeh.plotting import figure as bokeh_figure

from arviz_plots.backend.bokeh import scatter

pytestmark = [pytest.mark.usefixtures("check_skips"), pytest.mark.bokeh]


@pytest.fixture(scope="function")
def figure():
    return bokeh_figure()


def test_scatter(figure):
    x = np.array([0, 1, 2])
    y = np.array([0, 2, 1])

    scatter_obj = scatter(x, y, figure)

    assert len(figure.renderers) == 1
    assert figure.renderers[0] is scatter_obj
    assert np.array_equal(scatter_obj.data_source.data["x"], x)
    assert np.array_equal(scatter_obj.data_source.data["y"], y)


def test_scatter_args(figure):
    scatter_obj = scatter(
        [0, 1],
        [0, 0],
        figure,
        marker="circle",
        size=16,
        color="orange",
        alpha=0.5,
        width=2,
    )

    assert scatter_obj.glyph.size == 4
    assert scatter_obj.glyph.marker == "circle"
    assert scatter_obj.glyph.fill_color == "orange"
    assert scatter_obj.glyph.line_color == "orange"
    assert scatter_obj.glyph.fill_alpha == 0.5
    assert scatter_obj.glyph.line_alpha == 0.5
    assert scatter_obj.glyph.line_width == 2


def test_scatter_vertical_marker(figure):
    scatter_obj = scatter([0, 1], [0, 0], figure, marker="|")

    assert scatter_obj.glyph.marker == "dash"
    assert scatter_obj.glyph.angle == np.pi / 2
