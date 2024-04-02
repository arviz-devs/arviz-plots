# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots."""
import numpy as np
import pytest
from arviz_base import from_dict, load_arviz_data

from arviz_plots import plot_dist, plot_forest, plot_trace, plot_trace_dist, visuals

pytestmark = pytest.mark.usefixtures("clean_plots")
pytestmark = pytest.mark.usefixtures("check_skips")


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


@pytest.fixture(scope="module")
def datatree_sample(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=100)
    tau = rng.normal(size=100)
    theta = rng.normal(size=(100, 7))
    diverging = rng.choice([True, False], size=100, p=[0.1, 0.9])

    return from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy"]},
        sample_dims=["sample"],
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

    def test_plot_dist_sample(self, datatree_sample, backend):
        pc = plot_dist(datatree_sample, backend=backend, sample_dims="sample")
        assert not pc.aes["mu"]
        assert "kde" in pc.viz["mu"]
        assert "hierarchy" not in pc.viz["mu"].dims
        assert "hierarchy" in pc.viz["theta"].dims
        assert "hierarchy" not in pc.viz["mu"]["point_estimate"].dims
        assert "hierarchy" in pc.viz["theta"]["point_estimate"].dims

    def test_plot_dist_models(self, datatree, datatree2, backend):
        pc = plot_dist({"c": datatree, "n": datatree2}, backend=backend)
        assert "/mu" in pc.aes.groups
        assert "/mu" in pc.viz.groups
        assert "kde" in pc.viz["mu"].data_vars
        assert "hierarchy" not in pc.viz["mu"].dims
        assert "model" in pc.viz["mu"].dims

    def test_plot_trace(self, datatree, backend):
        pc = plot_trace(datatree, backend=backend)
        assert "chart" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert pc.viz["mu"].trace.shape == (4,)

    def test_plot_trace_sample(self, datatree_sample, backend):
        pc = plot_trace(datatree_sample, sample_dims="sample", backend=backend)
        assert "chart" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert pc.viz["mu"].trace.shape == ()

    @pytest.mark.parametrize("compact", (True, False))
    @pytest.mark.parametrize("combined", (True, False))
    def test_plot_trace_dist(self, datatree, backend, compact, combined):
        pc = plot_trace_dist(datatree, backend=backend, compact=compact, combined=combined)
        assert "chart" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "chain" in pc.viz["theta"]["trace"].dims
        if combined:
            assert "chain" not in pc.viz["theta"]["dist"].dims
        else:
            assert "chain" in pc.viz["theta"]["dist"].dims
        if compact:
            assert "hierarchy" not in pc.viz["theta"]["plot"].dims
        else:
            assert "hierarchy" in pc.viz["theta"]["plot"].dims

    @pytest.mark.parametrize("compact", (True, False))
    def test_plot_trace_dist_sample(self, datatree_sample, backend, compact):
        pc = plot_trace_dist(
            datatree_sample, backend=backend, sample_dims="sample", compact=compact
        )
        assert "chart" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        if compact:
            assert "hierarchy" not in pc.viz["theta"]["plot"].dims
        else:
            assert "hierarchy" in pc.viz["theta"]["plot"].dims

    @pytest.mark.parametrize("combined", (True, False))
    def test_plot_forest(self, datatree, backend, combined):
        pc = plot_forest(datatree, backend=backend, combined=combined)
        assert "plot" in pc.viz.data_vars
        assert all("y" in child.data_vars for child in pc.aes.children.values())

    def test_plot_forest_sample(self, datatree_sample, backend):
        pc = plot_forest(datatree_sample, backend=backend, sample_dims="sample")
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
        assert all("ess" in child.data_vars for child in pc.viz.children.values())

    def test_plot_forest_aes_labels_shading(self, backend):
        post = load_arviz_data("rugby_field").posterior.ds.sel(draw=slice(None, 100))
        for pseudo_dim in ("__variable__", "field", "team"):
            pc = plot_forest(
                post,
                pc_kwargs={"aes": {"color": [pseudo_dim]}},
                aes_map={"labels": ["color"]},
                shade_label=pseudo_dim,
                backend=backend,
            )
            assert "plot" in pc.viz.data_vars
            assert all("shade" in child.data_vars for child in pc.viz.children.values())
