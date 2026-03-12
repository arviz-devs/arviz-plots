# pylint: disable=no-self-use, redefined-outer-name, wrong-import-position
"""Tests specific to the matplotlib backend."""
import os

import numpy as np
import pytest

if os.environ.get("ARVIZ_REQUIRE_ALL_DEPS", False):
    import matplotlib  # pylint: disable=unused-import
else:
    pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Text
from matplotlib.collections import PolyCollection

from arviz_plots.backend.matplotlib import (
    line,
    scatter,
    text,
    vline,
    hline,
    step,
    title,
    xlabel,
    ylabel,
    hist,
    fill_between_y,
    ciliney,
    multiple_lines,
)

pytestmark = [
    pytest.mark.usefixtures("clean_plots"),
    pytest.mark.usefixtures("check_skips"),
    pytest.mark.matplotlib,
]


@pytest.fixture(scope="function")
def fig_ax():
    return plt.subplots()


class TestLine:
    """Tests for the line function."""

    def test_line(self, fig_ax):
        ax = fig_ax[1]
        line_obj = line([0, 1, 2], [0, 2, 1], ax)
        assert isinstance(line_obj, Line2D)
        assert line_obj.get_zorder() == 2
        assert len(ax.lines) == 1
        assert ax.lines[0] is line_obj

    def test_line_args(self, fig_ax):
        ax = fig_ax[1]
        line_obj = line([0, 1, 2], [0, 2, 1], ax, color="orange", width=2.2)
        assert line_obj.get_linewidth() == 2.2
        assert line_obj.get_color() == "orange"

    def test_line_linestyle(self, fig_ax):
        ax = fig_ax[1]
        line_obj = line([0, 1, 2], [0, 2, 1], ax, linestyle="--")
        assert line_obj.get_linestyle() == "--"

    def test_line_alpha(self, fig_ax):
        ax = fig_ax[1]
        line_obj = line([0, 1, 2], [0, 2, 1], ax, alpha=0.5)
        assert line_obj.get_alpha() == 0.5


class TestScatter:
    """Tests for the scatter function."""

    def test_scatter_basic(self, fig_ax):
        ax = fig_ax[1]
        scatter_obj = scatter([0, 1, 2], [0, 2, 1], ax)
        assert isinstance(scatter_obj, PathCollection)
        assert scatter_obj.get_zorder() == 2

    def test_scatter_color(self, fig_ax):
        ax = fig_ax[1]
        scatter_obj = scatter([0, 1, 2], [0, 2, 1], ax, color="red")
        # Check facecolor is set
        facecolors = scatter_obj.get_facecolors()
        assert len(facecolors) > 0

    def test_scatter_size(self, fig_ax):
        ax = fig_ax[1]
        scatter_obj = scatter([0, 1, 2], [0, 2, 1], ax, size=100)
        sizes = scatter_obj.get_sizes()
        assert all(s == 100 for s in sizes)

    def test_scatter_marker(self, fig_ax):
        ax = fig_ax[1]
        scatter_obj = scatter([0, 1, 2], [0, 2, 1], ax, marker="^")
        assert isinstance(scatter_obj, PathCollection)

    def test_scatter_alpha(self, fig_ax):
        ax = fig_ax[1]
        scatter_obj = scatter([0, 1, 2], [0, 2, 1], ax, alpha=0.3)
        assert scatter_obj.get_alpha() == 0.3


class TestText:
    """Tests for the text function."""

    def test_text_basic(self, fig_ax):
        ax = fig_ax[1]
        text_obj = text(0.5, 0.5, "hello", ax)
        assert isinstance(text_obj, Text)
        assert text_obj.get_text() == "hello"

    def test_text_position(self, fig_ax):
        ax = fig_ax[1]
        text_obj = text(0.2, 0.8, "test", ax)
        pos = text_obj.get_position()
        assert pos == (0.2, 0.8)

    def test_text_color(self, fig_ax):
        ax = fig_ax[1]
        text_obj = text(0.5, 0.5, "colored", ax, color="blue")
        assert text_obj.get_color() == "blue"

    def test_text_size(self, fig_ax):
        ax = fig_ax[1]
        text_obj = text(0.5, 0.5, "sized", ax, size=14)
        assert text_obj.get_fontsize() == 14


class TestVline:
    """Tests for the vline function."""

    def test_vline_basic(self, fig_ax):
        ax = fig_ax[1]
        vline_obj = vline(0.5, ax)
        assert isinstance(vline_obj, Line2D)
        # vline has zorder 0 by default
        assert vline_obj.get_zorder() == 0

    def test_vline_color(self, fig_ax):
        ax = fig_ax[1]
        vline_obj = vline(0.5, ax, color="green")
        assert vline_obj.get_color() == "green"

    def test_vline_linestyle(self, fig_ax):
        ax = fig_ax[1]
        vline_obj = vline(0.5, ax, linestyle=":")
        assert vline_obj.get_linestyle() == ":"


class TestHline:
    """Tests for the hline function."""

    def test_hline_basic(self, fig_ax):
        ax = fig_ax[1]
        hline_obj = hline(0.5, ax)
        assert isinstance(hline_obj, Line2D)
        assert hline_obj.get_zorder() == 2

    def test_hline_color(self, fig_ax):
        ax = fig_ax[1]
        hline_obj = hline(0.5, ax, color="purple")
        assert hline_obj.get_color() == "purple"

    def test_hline_width(self, fig_ax):
        ax = fig_ax[1]
        hline_obj = hline(0.5, ax, width=3.0)
        assert hline_obj.get_linewidth() == 3.0


class TestStep:
    """Tests for the step function."""

    def test_step_basic(self, fig_ax):
        ax = fig_ax[1]
        step_obj = step([0, 1, 2, 3], [0, 1, 0, 1], ax)
        assert isinstance(step_obj, Line2D)
        assert step_obj.get_zorder() == 2

    def test_step_color(self, fig_ax):
        ax = fig_ax[1]
        step_obj = step([0, 1, 2], [0, 1, 0], ax, color="cyan")
        assert step_obj.get_color() == "cyan"


class TestTitle:
    """Tests for the title function."""

    def test_title_basic(self, fig_ax):
        ax = fig_ax[1]
        title_obj = title("My Title", ax)
        assert isinstance(title_obj, Text)
        assert title_obj.get_text() == "My Title"

    def test_title_color(self, fig_ax):
        ax = fig_ax[1]
        title_obj = title("Colored Title", ax, color="red")
        assert title_obj.get_color() == "red"

    def test_title_size(self, fig_ax):
        ax = fig_ax[1]
        title_obj = title("Sized Title", ax, size=20)
        assert title_obj.get_fontsize() == 20


class TestLabels:
    """Tests for xlabel and ylabel functions."""

    def test_xlabel_basic(self, fig_ax):
        ax = fig_ax[1]
        xlabel_obj = xlabel("X Axis", ax)
        assert isinstance(xlabel_obj, Text)
        assert xlabel_obj.get_text() == "X Axis"

    def test_ylabel_basic(self, fig_ax):
        ax = fig_ax[1]
        ylabel_obj = ylabel("Y Axis", ax)
        assert isinstance(ylabel_obj, Text)
        assert ylabel_obj.get_text() == "Y Axis"

    def test_xlabel_color(self, fig_ax):
        ax = fig_ax[1]
        xlabel_obj = xlabel("X Axis", ax, color="blue")
        assert xlabel_obj.get_color() == "blue"

    def test_ylabel_size(self, fig_ax):
        ax = fig_ax[1]
        ylabel_obj = ylabel("Y Axis", ax, size=12)
        assert ylabel_obj.get_fontsize() == 12


class TestHist:
    """Tests for the hist function."""

    def test_hist_basic(self, fig_ax):
        ax = fig_ax[1]
        y = np.array([1, 2, 3, 2, 1])
        l_e = np.array([0, 1, 2, 3, 4])
        r_e = np.array([1, 2, 3, 4, 5])
        hist_obj = hist(y, l_e, r_e, ax)
        assert isinstance(hist_obj, PolyCollection)

    def test_hist_color(self, fig_ax):
        ax = fig_ax[1]
        y = np.array([1, 2, 3])
        l_e = np.array([0, 1, 2])
        r_e = np.array([1, 2, 3])
        hist_obj = hist(y, l_e, r_e, ax, color="orange")
        assert isinstance(hist_obj, PolyCollection)


class TestFillBetweenY:
    """Tests for the fill_between_y function."""

    def test_fill_between_y_basic(self, fig_ax):
        ax = fig_ax[1]
        x = np.array([0, 1, 2, 3])
        y_bottom = np.array([0, 0, 0, 0])
        y_top = np.array([1, 2, 1, 2])
        fill_obj = fill_between_y(x, y_bottom, y_top, ax)
        assert isinstance(fill_obj, PolyCollection)


class TestCiliney:
    """Tests for the ciliney function."""

    def test_ciliney_basic(self, fig_ax):
        ax = fig_ax[1]
        ciliney_obj = ciliney(0.5, 0.2, 0.8, ax)
        assert isinstance(ciliney_obj, Line2D)
        assert ciliney_obj.get_zorder() == 2

    def test_ciliney_color(self, fig_ax):
        ax = fig_ax[1]
        ciliney_obj = ciliney(0.5, 0.2, 0.8, ax, color="red")
        assert ciliney_obj.get_color() == "red"


class TestMultipleLines:
    """Tests for the multiple_lines function."""

    def test_multiple_lines_basic(self, fig_ax):
        ax = fig_ax[1]
        x = np.array([0, 1, 2])
        y = np.array([[0, 1, 0], [1, 0, 1], [0.5, 0.5, 0.5]])
        lines_obj = multiple_lines(x, y, ax)
        assert isinstance(lines_obj, LineCollection)
        assert lines_obj.get_zorder() == 2