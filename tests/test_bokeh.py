# pylint: disable=redefined-outer-name, wrong-import-position
"""Tests specific to the bokeh backend."""

import os

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


def test_vertical_marker_converts_area_to_length(figure):
    scatter_obj = scatter([0, 1], [0, 0], figure, marker="|", size=16)

    assert scatter_obj.glyph.size == 4
