# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots."""
import numpy as np
import pytest
from arviz_base import from_dict

from arviz_plots import plot_dist, plot_forest, plot_trace, visuals


@pytest.fixture(scope="module")
def datatree(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(4, 100))
    tau = rng.normal(size=(4, 100))
    theta = rng.normal(size=(4, 100, 7))
    diverging = rng.choice([True, False], size=(4, 100), p=[0.1, 0.9])

    return from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy"]},
    )


@pytest.fixture(scope="module")
def datatree2(seed=17):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(4, 100))
    tau = rng.normal(size=(4, 100))
    theta = rng.normal(size=(4, 100, 7))
    theta_t = rng.normal(size=(4, 100, 7))
    diverging = rng.choice([True, False], size=(4, 100), p=[0.1, 0.9])

    return from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau, "theta_t": theta_t},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy"], "theta_t": ["hierarchy"]},
    )


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

    def test_plot_forest(self, datatree, backend):
        pc = plot_forest(datatree, backend=backend)
        assert "plot" in pc.viz.data_vars

    def test_plot_forest_models(self, datatree, datatree2, backend):
        pc = plot_forest({"c": datatree, "n": datatree2}, backend=backend)
        assert "plot" in pc.viz.data_vars

    def test_plot_forest_extendable(self, datatree, backend):
        dt_aux = (
            datatree["posterior"]
            .expand_dims(column=3)
            .assign_coords(column=["labels", "forest", "ess"])
        )
        pc = plot_forest(dt_aux, combined=True, backend=backend)
        mock_ess = datatree["posterior"].ds.mean(("chain", "draw"))
        pc.map(visuals.scatter_x, "ess", data=mock_ess, coords={"column": "ess"}, color="blue")
        assert "plot" in pc.viz.data_vars
        assert pc.viz["plot"].sizes["column"] == 3

    def test_plot_forest_color_shading(self, datatree2, backend):
        pc = plot_forest(
            datatree2,
            pc_kwargs={"aes": {"color": ["__variable__"]}},
            aes_map={"labels": ["color"]},
            shade_label="hierarchy",
            backend=backend,
        )
        assert "plot" in pc.viz.data_vars
