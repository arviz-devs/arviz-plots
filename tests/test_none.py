"""Tests specific to the none backend."""

import numpy as np
import xarray as xr

from arviz_plots import visuals
from arviz_plots.backend.none import hexbin, histogram2d


def test_histogram2d():
    target = []
    artist = histogram2d(
        [0, 1, 3],
        [-1, 0, 2],
        [[1, 2], [3, 4]],
        target,
        alpha=0.4,
        vmin=0,
        vmax=5,
        custom_option=True,
    )

    assert target == [artist]
    assert artist["function"] == "histogram2d"
    assert artist["cmap"] == "viridis"
    assert artist["alpha"] == 0.4
    assert artist["vmin"] == 0
    assert artist["vmax"] == 5
    assert artist["custom_option"] is True
    assert np.array_equal(artist["x_edges"], [0, 1, 3])
    assert np.array_equal(artist["y_edges"], [-1, 0, 2])
    assert np.array_equal(artist["values"], [[1, 2], [3, 4]])


def test_hexbin():
    target = []
    x_vertices = np.array([[0, 1, 1, 0], [2, 3, 3, 2]])
    y_vertices = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    artist = hexbin(x_vertices, y_vertices, [2, 5], target, custom_option=True)

    assert target == [artist]
    assert artist["function"] == "hexbin"
    assert artist["cmap"] == "viridis"
    assert artist["custom_option"] is True
    assert np.array_equal(artist["x_vertices"], x_vertices)
    assert np.array_equal(artist["y_vertices"], y_vertices)
    assert np.array_equal(artist["values"], [2, 5])


def test_hexbin_visual_builds_shared_vertices():
    centers_x = np.array([0.0, 1.0, 0.5])
    centers_y = np.array([0.0, 0.0, 0.5])
    values = xr.Dataset(
        {
            "values": ("hexbin", [1, 2, 3]),
            "x_centers": ("hexbin", centers_x),
            "y_centers": ("hexbin", centers_y),
        }
    )

    artist = visuals.hexbin(values, [])
    vertices = np.stack((artist["x_vertices"], artist["y_vertices"]), axis=-1)

    assert vertices.shape == (3, 6, 2)
    assert np.allclose(vertices.mean(axis=1), np.column_stack((centers_x, centers_y)))
    assert np.allclose(vertices - vertices.mean(axis=1, keepdims=True), vertices[0] - centers_x[0])
