# pylint: disable=no-self-use, redefined-outer-name, wrong-import-position
"""Tests specific to the Bokeh backend."""

import os

import numpy as np
import pytest

if os.environ.get("ARVIZ_REQUIRE_ALL_DEPS", False):
    import bokeh  # noqa: F401  # pylint: disable=unused-import
else:
    pytest.importorskip("bokeh")

from bokeh import palettes as bp
from bokeh.models import GlyphRenderer, LinearColorMapper, Patches, Quad
from bokeh.plotting import figure as make_figure

from arviz_plots.backend.bokeh import hexbin, histogram2d

pytestmark = [pytest.mark.usefixtures("check_skips"), pytest.mark.bokeh]


@pytest.fixture(scope="function")
def figure():
    return make_figure()


def test_histogram2d(figure):
    x_edges = np.array([0, 1, 3])
    y_edges = np.array([-1, 0, 2, 5])
    values = np.arange(6).reshape(2, 3)

    renderer = histogram2d(
        x_edges,
        y_edges,
        values,
        figure,
        alpha=0.4,
        vmin=-1,
        vmax=8,
        line_width=2,
    )

    assert isinstance(renderer, GlyphRenderer)
    assert isinstance(renderer.glyph, Quad)
    assert np.array_equal(renderer.data_source.data["left"], [0, 0, 0, 1, 1, 1])
    assert np.array_equal(renderer.data_source.data["right"], [1, 1, 1, 3, 3, 3])
    assert np.array_equal(renderer.data_source.data["bottom"], [-1, 0, 2, -1, 0, 2])
    assert np.array_equal(renderer.data_source.data["top"], [0, 2, 5, 0, 2, 5])
    assert np.array_equal(renderer.data_source.data["values"], values.ravel())
    assert renderer.glyph.fill_alpha == 0.4
    assert renderer.glyph.line_width == 2
    mapper = renderer.glyph.fill_color.transform
    assert isinstance(mapper, LinearColorMapper)
    assert mapper.palette == bp.viridis(256)
    assert mapper.low == -1
    assert mapper.high == 8


def test_hexbin(figure):
    x_vertices = np.array([[0, 1, 1, 0], [2, 3, 3, 2]])
    y_vertices = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    values = np.array([2, 5])

    renderer = hexbin(
        x_vertices,
        y_vertices,
        values,
        figure,
        alpha=0.6,
        vmin=1,
        vmax=7,
        line_width=2,
    )

    assert isinstance(renderer, GlyphRenderer)
    assert isinstance(renderer.glyph, Patches)
    assert np.array_equal(renderer.data_source.data["xs"], x_vertices)
    assert np.array_equal(renderer.data_source.data["ys"], y_vertices)
    assert np.array_equal(renderer.data_source.data["values"], values)
    assert renderer.glyph.fill_alpha == 0.6
    assert renderer.glyph.line_width == 2
    mapper = renderer.glyph.fill_color.transform
    assert isinstance(mapper, LinearColorMapper)
    assert mapper.low == 1
    assert mapper.high == 7


@pytest.mark.parametrize("values", [np.full((2, 2), 4.0), np.full((2, 2), np.nan)])
def test_histogram2d_degenerate_color_limits(figure, values):
    renderer = histogram2d([0, 1, 2], [0, 1, 2], values, figure)
    mapper = renderer.glyph.fill_color.transform

    assert np.isfinite([mapper.low, mapper.high]).all()
    assert mapper.low < mapper.high


@pytest.mark.parametrize("values", [np.full(2, 4.0), np.full(2, np.nan)])
def test_hexbin_degenerate_color_limits(figure, values):
    renderer = hexbin(
        [[0, 1, 1, 0], [2, 3, 3, 2]],
        [[0, 0, 1, 1], [1, 1, 2, 2]],
        values,
        figure,
    )
    mapper = renderer.glyph.fill_color.transform

    assert np.isfinite([mapper.low, mapper.high]).all()
    assert mapper.low < mapper.high
