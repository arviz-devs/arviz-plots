# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots."""
import numpy as np
import pandas as pd
import pytest
from arviz_base import from_dict
from scipy.stats import halfnorm, norm

from arviz_plots import (
    add_bands,
    add_lines,
    plot_autocorr,
    plot_bf,
    plot_compare,
    plot_dist,
    plot_energy,
    plot_ess,
    plot_ess_evolution,
    plot_forest,
    plot_mcse,
    plot_ppc_dist,
    plot_prior_posterior,
    plot_psense_dist,
    plot_rank_dist,
    plot_ridge,
    plot_trace,
    plot_trace_dist,
    visuals,
)

pytestmark = [
    pytest.mark.usefixtures("clean_plots"),
    pytest.mark.usefixtures("check_skips"),
    pytest.mark.usefixtures("no_artist_kwargs"),
]


def generate_base_data(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(loc=1, size=(4, 100))
    tau = np.exp(rng.normal(size=(4, 100)))
    theta_orig = rng.uniform(size=7)
    theta = rng.normal(theta_orig[None, None, :], scale=1, size=(4, 100, 7))
    idxs = rng.choice(np.arange(7), size=29)
    x = np.linspace(0, 1, 29)
    obs = rng.normal(loc=x + theta_orig[idxs], scale=3)
    log_lik = norm(mu[:, :, None] * x[None, None, :] + theta[:, :, idxs], tau[:, :, None]).logpdf(
        obs[None, None, :]
    )
    log_lik = log_lik / log_lik.var()
    log_mu_prior = norm(0, 3).logpdf(mu)
    log_tau_prior = halfnorm(scale=5).logpdf(tau)
    log_theta_prior = norm(0, 1).logpdf(theta)
    prior_predictive = rng.normal(size=(1, 100, 7))
    posterior_predictive = rng.normal(size=(4, 100, 7))
    diverging = rng.choice([True, False], size=(4, 100), p=[0.1, 0.9])
    mu_prior = norm(0, 3).rvs(size=(1, 500), random_state=rng)
    tau_prior = halfnorm(0, 5).rvs(size=(1, 500), random_state=rng)
    theta_prior = norm(0, 1).rvs(size=(1, 500, 7), random_state=rng)
    energy = rng.normal(loc=50, scale=10, size=(4, 100))

    return {
        "posterior": {"mu": mu, "theta": theta, "tau": tau},
        "observed_data": {"y": obs},
        "log_likelihood": {"y": log_lik},
        "log_prior": {"mu": log_mu_prior, "theta": log_theta_prior, "tau": log_tau_prior},
        "prior_predictive": {"y": prior_predictive},
        "posterior_predictive": {"y": posterior_predictive},
        "sample_stats": {"diverging": diverging, "energy": energy},
        "prior": {"mu": mu_prior, "theta": theta_prior, "tau": tau_prior},
    }


@pytest.fixture(scope="module")
def datatree(seed=31):
    base_data = generate_base_data(seed)

    return from_dict(
        base_data,
        dims={"theta": ["hierarchy"], "y": ["obs_dim"]},
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
def datatree_4d(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(4, 100))
    theta = rng.normal(size=(4, 100, 5))
    eta = rng.normal(size=(4, 100, 5, 3))
    diverging = rng.choice([True, False], size=(4, 100), p=[0.1, 0.9])
    obs = rng.normal(size=(5, 3))
    prior_predictive = rng.normal(size=(1, 100, 5, 3))
    posterior_predictive = rng.normal(size=(4, 100, 5, 3))

    return from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "eta": eta},
            "observed_data": {"obs": obs},
            "prior_predictive": {"obs": prior_predictive},
            "posterior_predictive": {"obs": posterior_predictive},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy"], "eta": ["hierarchy", "group"]},
    )


@pytest.fixture(scope="module")
def datatree_sample(seed=31):
    base_data = generate_base_data(seed)

    return from_dict(
        {
            group: {
                key: values[0] if group != "observed_data" else values
                for key, values in group_dict.items()
            }
            for group, group_dict in base_data.items()
        },
        dims={"theta": ["hierarchy"]},
        sample_dims=["sample"],
    )


@pytest.fixture(scope="module")
def cmp():
    return pd.DataFrame(
        {
            "elpd": [-4.5, -14.3, -16.2],
            "p": [2.6, 2.3, 2.1],
            "elpd_diff": [0, 9.7, 11.3],
            "weight": [0.9, 0.1, 0],
            "se": [2.3, 2.7, 2.3],
            "dse": [0, 2.7, 2.3],
            "warning": [False, False, False],
        },
        index=["Model B", "Model A", "Model C"],
    )


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
class TestPlots:  # pylint: disable=too-many-public-methods
    @pytest.mark.parametrize("kind", ["kde", "hist", "ecdf"])
    def test_plot_dist(self, datatree, backend, kind):
        pc = plot_dist(datatree, backend=backend, kind=kind)
        assert not pc.aes
        assert "mu" in pc.viz[kind].data_vars
        artists = ("plot", kind, "credible_interval", "point_estimate")
        assert all("hierarchy" not in pc.viz[artist]["mu"].dims for artist in artists)
        assert all("hierarchy" in pc.viz[artist]["theta"].dims for artist in artists)

    def test_plot_dist_step_hist(self, datatree, backend):
        plot_kwargs = {"hist": {"step": True}}
        pc = plot_dist(datatree, backend=backend, kind="hist", plot_kwargs=plot_kwargs)
        assert not pc.aes
        assert "mu" in pc.viz["hist"].data_vars
        artists = ("plot", "hist", "credible_interval", "point_estimate")
        assert all("hierarchy" not in pc.viz[artist]["mu"].dims for artist in artists)
        assert all("hierarchy" in pc.viz[artist]["theta"].dims for artist in artists)

    @pytest.mark.parametrize("kind", ["kde", "hist", "ecdf"])
    def test_plot_dist_sample(self, datatree_sample, backend, kind):
        pc = plot_dist(datatree_sample, backend=backend, sample_dims="sample", kind=kind)
        assert not pc.aes
        assert "mu" in pc.viz[kind].data_vars
        artists = ("plot", kind, "credible_interval", "point_estimate")
        assert all("hierarchy" not in pc.viz[artist]["mu"].dims for artist in artists)
        assert all("hierarchy" in pc.viz[artist]["theta"].dims for artist in artists)

    def test_plot_dist_sample_step_hist(self, datatree_sample, backend):
        plot_kwargs = {"hist": {"step": True}}
        pc = plot_dist(
            datatree_sample,
            backend=backend,
            sample_dims="sample",
            kind="hist",
            plot_kwargs=plot_kwargs,
        )
        assert not pc.aes
        assert "mu" in pc.viz["hist"].data_vars
        artists = ("plot", "hist", "credible_interval", "point_estimate")
        assert all("hierarchy" not in pc.viz[artist]["mu"].dims for artist in artists)
        assert all("hierarchy" in pc.viz[artist]["theta"].dims for artist in artists)

    @pytest.mark.parametrize("kind", ["kde"])
    def test_plot_dist_models(self, datatree, datatree2, backend, kind):
        pc = plot_dist({"c": datatree, "n": datatree2}, backend=backend, kind=kind)
        assert "/color" in pc.aes.groups
        assert tuple(pc.aes["color"].dims) == ("model",)
        assert kind in pc.viz.children
        assert "mu" in pc.viz[kind].data_vars
        assert "hierarchy" not in pc.viz[kind]["mu"].dims
        assert "model" in pc.viz[kind]["mu"].dims

    def test_plot_trace(self, datatree, backend):
        pc = plot_trace(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert pc.viz["trace"]["mu"].shape == (4,)

    def test_plot_trace_sample(self, datatree_sample, backend):
        pc = plot_trace(datatree_sample, sample_dims="sample", backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert pc.viz["trace"]["mu"].shape == ()

    @pytest.mark.parametrize("compact", (True, False))
    @pytest.mark.parametrize("combined", (True, False))
    def test_plot_trace_dist(self, datatree, backend, compact, combined):
        kind = "kde"
        pc = plot_trace_dist(datatree, backend=backend, compact=compact, combined=combined)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "chain" in pc.viz["trace"]["theta"].dims
        if combined:
            assert "chain" not in pc.viz[kind]["theta"].dims
        else:
            assert "chain" in pc.viz[kind]["theta"].dims
        if compact:
            assert "hierarchy" not in pc.viz["plot"]["theta"].dims
        else:
            assert "hierarchy" in pc.viz["plot"]["theta"].dims

    @pytest.mark.parametrize("compact", (True, False))
    def test_plot_trace_dist_sample(self, datatree_sample, backend, compact):
        pc = plot_trace_dist(
            datatree_sample, backend=backend, sample_dims="sample", compact=compact
        )
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        if compact:
            assert "hierarchy" not in pc.viz["plot"]["theta"].dims
        else:
            assert "hierarchy" in pc.viz["plot"]["theta"].dims

    @pytest.mark.parametrize("compact", (True, False))
    @pytest.mark.parametrize("combined", (True, False))
    def test_plot_rank_dist(self, datatree, backend, compact, combined):
        kind = "kde"
        pc = plot_rank_dist(datatree, backend=backend, compact=compact, combined=combined)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        if combined:
            assert "chain" not in pc.viz[kind]["theta"].dims
        else:
            assert "chain" in pc.viz[kind]["theta"].dims
        if compact:
            assert "hierarchy" not in pc.viz["plot"]["theta"].dims
        else:
            assert "hierarchy" in pc.viz["plot"]["theta"].dims

    @pytest.mark.parametrize("combined", (True, False))
    def test_plot_forest(self, datatree, backend, combined):
        pc = plot_forest(datatree, backend=backend, combined=combined)
        assert "plot" in pc.viz.data_vars
        assert "/y" in pc.aes.groups
        assert all(var_name in pc.aes["y"].data_vars for var_name in datatree.posterior.data_vars)

    def test_plot_forest_sample(self, datatree_sample, backend):
        pc = plot_forest(datatree_sample, backend=backend, sample_dims="sample")
        assert "plot" in pc.viz.data_vars

    def test_plot_forest_models(self, datatree, datatree2, backend):
        pc = plot_forest({"c": datatree, "n": datatree2}, backend=backend)
        assert "plot" in pc.viz.data_vars

    def test_plot_forest_extendable(self, datatree, backend):
        dt_aux = (
            datatree["posterior"]
            .dataset.expand_dims(column=3)
            .assign_coords(column=["labels", "forest", "ess"])
        )
        pc = plot_forest(dt_aux, combined=True, backend=backend)
        mock_ess = datatree["posterior"].ds.mean(("chain", "draw"))
        pc.map(visuals.scatter_x, "ess", data=mock_ess, coords={"column": "ess"}, color="blue")
        assert "plot" in pc.viz.data_vars
        assert pc.viz["plot"].sizes["column"] == 3
        assert "ess" in pc.viz.children

    @pytest.mark.parametrize("pseudo_dim", ("__variable__", "hierarchy", "group"))
    def test_plot_forest_aes_labels_shading(self, backend, datatree_4d, pseudo_dim):
        pc = plot_forest(
            datatree_4d,
            pc_kwargs={"aes": {"color": [pseudo_dim]}},
            aes_map={"labels": ["color"]},
            shade_label=pseudo_dim,
            backend=backend,
        )
        assert "plot" in pc.viz.data_vars
        assert "shade" in pc.viz.children
        if pseudo_dim != "__variable__":
            assert pc.aes["alpha"]["neutral_element"].item() == 0
            assert 0 in pc.aes["alpha"]["mapping"].values
            assert pseudo_dim in pc.viz["shade"].dims

    @pytest.mark.parametrize("combined", (True, False))
    def test_plot_ridge(self, datatree, backend, combined):
        pc = plot_ridge(datatree, backend=backend, combined=combined)
        assert "plot" in pc.viz.data_vars
        assert "/y" in pc.aes.groups
        assert all(var_name in pc.aes["y"].data_vars for var_name in datatree.posterior.data_vars)
        assert "edge" in pc.viz.children
        assert "mu" in pc.viz["edge"]
        assert "hierarchy" not in pc.viz["edge"]["mu"].dims
        assert "hierarchy" in pc.viz["edge"]["theta"].dims

    def test_plot_ridge_sample(self, datatree_sample, backend):
        pc = plot_ridge(datatree_sample, backend=backend, sample_dims="sample")
        assert "plot" in pc.viz.data_vars
        assert "edge" in pc.viz.children
        assert "mu" in pc.viz["edge"]
        assert "hierarchy" not in pc.viz["edge"]["mu"].dims
        assert "hierarchy" in pc.viz["edge"]["theta"].dims

    def test_plot_ridge_models(self, datatree, datatree2, backend):
        pc = plot_ridge({"c": datatree, "n": datatree2}, backend=backend)
        assert "plot" in pc.viz.data_vars
        assert "/color" in pc.aes.groups
        assert "/edge" in pc.viz.groups
        assert "mu" in pc.viz["edge"].data_vars
        assert "hierarchy" not in pc.viz["edge"]["mu"].dims
        assert "hierarchy" in pc.viz["edge"]["theta"].dims
        assert "model" in pc.viz["edge"]["mu"].dims

    def test_plot_ridge_extendable(self, datatree, backend):
        dt_aux = (
            datatree["posterior"]
            .dataset.expand_dims(column=3)
            .assign_coords(column=["labels", "ridge", "ess"])
        )
        pc = plot_ridge(dt_aux, combined=True, backend=backend)
        mock_ess = datatree["posterior"].ds.mean(("chain", "draw"))
        pc.map(visuals.scatter_x, "ess", data=mock_ess, coords={"column": "ess"}, color="blue")
        assert "plot" in pc.viz.data_vars
        assert pc.viz["plot"].sizes["column"] == 3
        assert "ess" in pc.viz.children

    @pytest.mark.parametrize("pseudo_dim", ("__variable__", "hierarchy", "group"))
    def test_plot_ridge_aes_labels_shading(self, backend, datatree_4d, pseudo_dim):
        pc = plot_forest(
            datatree_4d,
            pc_kwargs={"aes": {"color": [pseudo_dim]}},
            aes_map={"labels": ["color"]},
            shade_label=pseudo_dim,
            backend=backend,
        )
        assert "plot" in pc.viz.data_vars
        assert "shade" in pc.viz.children
        if pseudo_dim != "__variable__":
            assert pc.aes["alpha"]["neutral_element"].item() == 0
            assert 0 in pc.aes["alpha"]["mapping"].values
            assert pseudo_dim in pc.viz["shade"].dims

    def test_plot_compare(self, cmp, backend):
        pc = plot_compare(cmp, backend=backend)
        assert "plot" in pc.viz.data_vars

    def test_plot_compare_kwargs(self, cmp, backend):
        pc = plot_compare(
            cmp,
            plot_kwargs={
                "shade": {"color": "black", "alpha": 0.2},
                "error_bar": {"color": "gray"},
                "point_estimate": {"color": "red", "marker": "|"},
            },
            pc_kwargs={"plot_grid_kws": {"figsize": (1000, 200), "figsize_units": "dots"}},
            backend=backend,
        )
        assert "plot" in pc.viz.data_vars

    def test_plot_ess(self, datatree, backend):
        pc = plot_ess(datatree, backend=backend, rug=True)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["ess"]
        assert "mu" in pc.viz["min_ess"]
        assert "mu" in pc.viz["title"]
        assert "mu" in pc.viz["rug"]
        assert all("hierarchy" not in child["mu"].dims for child in pc.viz.children.values())
        assert all("hierarchy" in child["theta"].dims for child in pc.viz.children.values())
        assert "chain" in pc.viz["rug"]["mu"].dims  # checking rug artist overlay
        # checking aesthetics
        assert "mapping" in pc.aes["overlay"]
        assert "chain" in pc.aes["overlay"]["mapping"].dims

    def test_plot_ess_sample(self, datatree_sample, backend):
        pc = plot_ess(datatree_sample, backend=backend, rug=True, sample_dims="sample")
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["ess"]
        assert "mu" in pc.viz["min_ess"]
        assert "mu" in pc.viz["title"]
        assert "mu" in pc.viz["rug"]
        assert all("hierarchy" not in child["mu"].dims for child in pc.viz.children.values())
        assert all("hierarchy" in child["theta"].dims for child in pc.viz.children.values())
        assert pc.viz["rug"]["mu"].shape == ()  # 0 chains here, so no overlay

    def test_plot_ess_models(self, datatree, datatree2, backend):
        pc = plot_ess({"c": datatree, "n": datatree2}, backend=backend, rug=False)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["ess"]
        assert "mu" in pc.viz["min_ess"]
        assert "mu" in pc.viz["title"]
        assert "rug" not in pc.viz.children
        assert all("hierarchy" not in child["mu"].dims for child in pc.viz.children.values())
        assert all("hierarchy" in child["theta"].dims for child in pc.viz.children.values())
        assert "model" in pc.viz["ess"]["mu"].dims
        # checking aesthetics
        assert "/color" in pc.aes.groups
        assert "model" in pc.aes["color"].dims
        assert "/x" in pc.aes.groups

    def test_plot_ess_evolution(self, datatree, backend):
        pc = plot_ess_evolution(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["ess_bulk"]
        assert "mu" in pc.viz["ess_tail"]
        assert "mu" in pc.viz["ess_bulk_line"]
        assert "mu" in pc.viz["ess_tail_line"]
        assert "mu" in pc.viz["min_ess"]
        assert "mu" in pc.viz["title"]
        assert all("hierarchy" not in child["mu"].dims for child in pc.viz.children.values())
        assert all("hierarchy" in child["theta"].dims for child in pc.viz.children.values())

    def test_plot_ess_evolution_sample(
        self, datatree_sample, backend
    ):  # pylint: disable=unused-argument
        pc = plot_ess_evolution(datatree_sample, sample_dims="sample")
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["ess_bulk"]
        assert "mu" in pc.viz["ess_tail"]
        assert "mu" in pc.viz["ess_bulk_line"]
        assert "mu" in pc.viz["ess_tail_line"]
        assert "mu" in pc.viz["min_ess"]
        assert "mu" in pc.viz["title"]
        assert all("hierarchy" not in child["mu"].dims for child in pc.viz.children.values())
        assert all("hierarchy" in child["theta"].dims for child in pc.viz.children.values())

    @pytest.mark.parametrize("kind", ["kde", "ecdf", "hist"])
    def test_plot_ppc_dist(self, datatree, kind, backend):
        pc = plot_ppc_dist(datatree, kind=kind, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "/overlay_ppc" in pc.aes.groups
        assert "y" in pc.viz[kind]
        assert "y" in pc.viz["observed_density"]

    def test_plot_psense_dist(self, datatree, backend):
        pc = plot_psense_dist(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "component_group" in pc.viz["plot"]["mu"].dims
        assert "alpha" not in pc.viz["plot"]["mu"].dims
        assert "mu" in pc.viz["credible_interval"]
        assert "component_group" in pc.viz["credible_interval"]["mu"].dims
        assert "alpha" in pc.viz["credible_interval"]["mu"].dims
        assert "hierarchy" in pc.viz["credible_interval"]["theta"].dims

    def test_plot_psense_dist_sample(self, datatree_sample, backend):
        pc = plot_psense_dist(datatree_sample, backend=backend, sample_dims="sample")
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "component_group" in pc.viz["plot"]["mu"].dims
        assert "alpha" not in pc.viz["plot"]["mu"].dims
        assert "mu" in pc.viz["credible_interval"]
        assert "component_group" in pc.viz["credible_interval"]["mu"].dims
        assert "alpha" in pc.viz["credible_interval"]["mu"].dims
        assert "hierarchy" in pc.viz["credible_interval"]["theta"].dims

    def test_plot_bf(self, datatree, backend):
        pc = plot_bf(datatree, var_names="mu", backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "group" in pc.viz["kde"].coords
        assert "/color" in pc.aes.groups
        assert "BF_type" in pc.aes["bf_aes"].coords

    def test_plot_energy_dist(self, datatree, backend):
        pc = plot_energy(datatree, backend=backend)
        assert pc is not None
        assert hasattr(pc, "viz")
        assert "/kde" in pc.viz.groups
        assert "energy" in pc.viz["kde"]
        assert "energy" in pc.viz["kde"].coords
        kde_values = pc.viz["kde"]["energy_"].values
        assert kde_values.size > 0
        assert "component_group" not in pc.viz["kde"]["energy_"].dims
        assert "alpha" not in pc.viz["kde"]["energy_"].dims
        energy_coords = pc.viz["kde"]["energy_"].coords["energy"].values
        assert "marginal" in energy_coords
        assert "transition" in energy_coords

    def test_plot_energy_dist_sample(self, datatree_sample, backend):
        pc = plot_energy(datatree_sample, backend=backend)
        assert pc is not None
        assert hasattr(pc, "viz")
        assert "/kde" in pc.viz.groups
        assert "energy" in pc.viz["kde"]
        assert "energy" in pc.viz["kde"].coords
        kde_values = pc.viz["kde"]["energy_"].values
        assert kde_values.size > 0
        assert "component_group" not in pc.viz["kde"]["energy_"].dims
        assert "alpha" not in pc.viz["kde"]["energy_"].dims
        energy_coords = pc.viz["kde"]["energy_"].coords["energy"].values
        assert "marginal" in energy_coords
        assert "transition" in energy_coords

    def test_plot_prior_posterior(self, datatree, backend):
        pc = plot_prior_posterior(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "group" not in pc.viz["plot"].coords
        assert "group" in pc.viz["kde"].coords

    def test_autocorr(self, datatree, backend):
        pc = plot_autocorr(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "hierarchy" not in pc.viz["lines"]["mu"].dims
        assert "hierarchy" in pc.viz["lines"]["theta"].dims

    def test_plot_mcse(self, datatree, backend):
        pc = plot_mcse(datatree, backend=backend, rug=True)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["mcse"]
        assert "mu" in pc.viz["title"]
        assert "mu" in pc.viz["rug"]
        assert "hierarchy" not in pc.viz["mcse"]["mu"].dims
        assert "hierarchy" in pc.viz["mcse"]["theta"].dims
        assert "chain" in pc.viz["rug"]["mu"].dims  # checking rug artist overlay
        # checking aesthetics
        assert "/overlay" in pc.aes.groups  # overlay of chains

    def test_plot_mcse_sample(self, datatree_sample, backend):
        pc = plot_mcse(datatree_sample, backend=backend, rug=True, sample_dims="sample")
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["mcse"]
        assert "mu" in pc.viz["title"]
        assert "mu" in pc.viz["rug"]
        assert "hierarchy" not in pc.viz["mcse"]["mu"].dims
        assert "hierarchy" in pc.viz["mcse"]["theta"].dims
        assert pc.viz["rug"]["mu"].shape == ()  # 0 chains here, so no overlay

    def test_plot_mcse_models(self, datatree, datatree2, backend):
        pc = plot_mcse({"c": datatree, "n": datatree2}, backend=backend, rug=False)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["mcse"]
        assert "mu" in pc.viz["title"]
        assert "rug" not in pc.viz.children
        assert "hierarchy" not in pc.viz["mcse"]["mu"].dims
        assert "hierarchy" in pc.viz["mcse"]["theta"].dims
        assert "model" in pc.viz["mcse"]["mu"].dims
        # checking aesthetics
        assert "/color" in pc.aes.groups
        assert "model" in pc.aes["color"].dims
        assert "/x" in pc.aes.groups

    def test_add_references_scalar(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        add_lines(pc, 0)
        assert "mu" in pc.viz["ref_line"]
        assert "ref_dim" not in pc.viz["ref_line"]["mu"].dims

    def test_add_references_array(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        add_lines(pc, [0, 1])
        assert "mu" in pc.viz["ref_line"]
        assert "ref_dim" in pc.viz["ref_line"]["mu"].dims

    def test_add_references_dict(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        add_lines(pc, {"mu": [0, 1]})
        assert "mu" in pc.viz["ref_line"]
        assert "theta" not in pc.viz["ref_line"]
        assert "ref_dim" in pc.viz["ref_line"]["mu"].dims

    def test_add_references_ds(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        add_lines(
            pc,
            datatree.posterior.dataset.quantile((0.1, 0.5, 0.9), dim=["chain", "draw"]),
            ref_dim="quantile",
        )
        assert "mu" in pc.viz["ref_line"]
        assert "theta" in pc.viz["ref_line"]
        assert "ref_dim" not in pc.viz["ref_line"].dims
        assert "quantile" in pc.viz["ref_line"].dims

    def test_add_references_aes(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        add_lines(pc, [0, 1], aes_map={"ref_line": ["color"]})
        assert "mu" in pc.viz["ref_line"].data_vars
        assert "ref_dim" in pc.viz["ref_line"]["mu"].dims
        assert "/color" in pc.aes.groups
        assert "ref_dim" in pc.aes["color"].dims

    def test_add_bands_array(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        add_bands(pc, [(0, 1), (2, 4)])
        assert "mu" in pc.viz["ref_band"]
        assert "ref_dim" in pc.viz["ref_band"]["mu"].dims

    def test_add_bands_dict(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        add_bands(pc, {"mu": [(0, 1)]})
        assert "mu" in pc.viz["ref_band"]
        assert "theta" not in pc.viz["ref_band"]
        assert "ref_dim" in pc.viz["ref_band"]["mu"].dims

    def test_add_bands_aes(self, datatree, backend):
        pc = plot_dist(datatree, backend=backend)
        add_bands(pc, [(0, 1), (2, 5)], aes_map={"ref_band": ["color"]})
        assert "mu" in pc.viz["ref_band"].data_vars
        assert "ref_dim" in pc.viz["ref_band"]["mu"].dims
        assert "/color" in pc.aes.groups
        assert "ref_dim" in pc.aes["color"].dims
