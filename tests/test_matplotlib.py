# pylint: disable=no-self-use, redefined-outer-name, wrong-import-position
"""Tests specific to the matplotlib backend."""
import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D

from arviz_plots.backend.matplotlib import line

pytestmark = [
    pytest.mark.usefixtures("clean_plots"),
    pytest.mark.usefixtures("check_skips"),
    pytest.mark.matplotlib,
]


@pytest.fixture(scope="function")
def fig_ax():
    return plt.subplots()


def test_line(fig_ax):
    ax = fig_ax[1]
    line_obj = line([0, 1, 2], [0, 2, 1], ax)
    assert isinstance(line_obj, Line2D)
    assert line_obj.get_zorder() == 2
    assert len(ax.lines) == 1
    assert ax.lines[0] is line_obj


def test_line_args(fig_ax):
    ax = fig_ax[1]
    line_obj = line([0, 1, 2], [0, 2, 1], ax, color="orange", width=2.2)
    assert line_obj.get_linewidth() == 2.2
    assert line_obj.get_color() == "orange"
