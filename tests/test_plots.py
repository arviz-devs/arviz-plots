# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots."""
import numpy as np
import pytest
from arviz_base import from_dict

from arviz_plots import plot_dist, plot_trace


@pytest.fixture(scope="module")
def datatree(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(4, 100))
    theta = rng.normal(size=(4, 100, 7))

    return from_dict({"posterior": {"mu": mu, "theta": theta}}, dims={"theta": ["hierarchy"]})


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh"])
class TestPlots:
    def test_plot_dist(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        assert not pc.aes["mu"]
        assert "kde" in pc.viz["mu"]
        assert "hierarchy" not in pc.viz["mu"].dims
        assert "hierarchy" in pc.viz["theta"].dims
        assert "hierarchy" not in pc.viz["mu"]["point_estimate"].dims
        assert "hierarchy" in pc.viz["theta"]["point_estimate"].dims

    def test_plot_trace(self, datatree, backend):
        pc = plot_trace(datatree, var_names=["mu"], backend=backend)
        assert "/mu" in pc.aes.groups
        assert "/theta" not in pc.aes.groups
        assert pc.viz["mu"].trace.shape == (4,)
