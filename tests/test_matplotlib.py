# pylint: disable=no-self-use, redefined-outer-name, wrong-import-position
"""Tests specific to the matplotlib backend."""

import os

import numpy as np
import pytest

if os.environ.get("ARVIZ_REQUIRE_ALL_DEPS", False):
    import matplotlib  # noqa: F401  # pylint: disable=unused-import
else:
    pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, QuadMesh
from matplotlib.lines import Line2D

from arviz_plots.backend.matplotlib import hexbin, histogram2d, line

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


def test_histogram2d(fig_ax):
    ax = fig_ax[1]
    x_edges = np.array([0, 1, 3])
    y_edges = np.array([-1, 0, 2, 5])
    values = np.arange(6).reshape(2, 3)

    artist = histogram2d(
        x_edges,
        y_edges,
        values,
        ax,
        alpha=0.4,
        vmin=-1,
        vmax=8,
        rasterized=True,
    )

    assert isinstance(artist, QuadMesh)
    assert artist.get_cmap().name == "viridis"
    assert artist.get_alpha() == 0.4
    assert artist.get_clim() == (-1, 8)
    assert artist.get_rasterized()
    assert np.array_equal(artist.get_coordinates()[0, :, 0], x_edges)
    assert np.array_equal(artist.get_coordinates()[:, 0, 1], y_edges)
    assert np.array_equal(artist.get_array(), values.T)


def test_hexbin(fig_ax):
    ax = fig_ax[1]
    x_vertices = np.array([[0, 1, 1, 0], [2, 3, 3, 2]])
    y_vertices = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    values = np.array([2, 5])

    artist = hexbin(
        x_vertices,
        y_vertices,
        values,
        ax,
        alpha=0.6,
        vmin=1,
        vmax=7,
        rasterized=True,
    )

    assert isinstance(artist, PolyCollection)
    assert artist.get_cmap().name == "viridis"
    assert artist.get_alpha() == 0.6
    assert artist.get_clim() == (1, 7)
    assert artist.get_rasterized()
    assert np.array_equal(artist.get_array(), values)
    assert np.array_equal(artist.get_paths()[0].vertices[:4, 0], x_vertices[0])
    assert np.array_equal(artist.get_paths()[0].vertices[:4, 1], y_vertices[0])


@pytest.mark.parametrize("values", [np.full((2, 2), 4.0), np.full((2, 2), np.nan)])
def test_histogram2d_degenerate_color_limits(fig_ax, values):
    artist = histogram2d([0, 1, 2], [0, 1, 2], values, fig_ax[1])

    low, high = artist.get_clim()
    assert np.isfinite([low, high]).all()
    assert low < high


@pytest.mark.parametrize("values", [np.full(2, 4.0), np.full(2, np.nan)])
def test_hexbin_degenerate_color_limits(fig_ax, values):
    artist = hexbin(
        [[0, 1, 1, 0], [2, 3, 3, 2]],
        [[0, 0, 1, 1], [1, 1, 2, 2]],
        values,
        fig_ax[1],
    )

    low, high = artist.get_clim()
    assert np.isfinite([low, high]).all()
    assert low < high
