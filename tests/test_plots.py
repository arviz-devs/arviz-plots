# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots."""
import pytest

from arviz_plots import (
    add_bands,
    add_lines,
    plot_autocorr,
    plot_bf,
    plot_compare,
    plot_convergence_dist,
    plot_dist,
    plot_ecdf_pit,
    plot_energy,
    plot_ess,
    plot_ess_evolution,
    plot_forest,
    plot_loo_pit,
    plot_mcse,
    plot_pair,
    plot_pair_focus,
    plot_parallel,
    plot_ppc_dist,
    plot_ppc_pava,
    plot_ppc_pit,
    plot_ppc_rootogram,
    plot_ppc_tstat,
    plot_prior_posterior,
    plot_psense_dist,
    plot_psense_quantities,
    plot_rank,
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


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
class TestPlots:  # pylint: disable=too-many-public-methods
    def test_autocorr(self, datatree, backend):
        pc = plot_autocorr(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "hierarchy" not in pc.viz["lines"]["mu"].dims
        assert "hierarchy" in pc.viz["lines"]["theta"].dims

    def test_plot_bf(self, datatree, backend):
        pc = plot_bf(datatree, var_names="mu", backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "group" in pc.viz["dist"].coords
        assert "/color" in pc.aes.groups
        assert "BF_type" in pc.aes["bf_aes"].coords

    def test_plot_compare(self, cmp, backend):
        pc = plot_compare(cmp, backend=backend)
        assert "plot" in pc.viz.data_vars

    def test_plot_compare_kwargs(self, cmp, backend):
        pc = plot_compare(
            cmp,
            visuals={
                "error_bar": {"color": "gray"},
                "point_estimate": {"color": "red", "marker": "|"},
                "ref_band": {"color": "blue", "alpha": 0.2},
                "similar_line": {"color": "black"},
            },
            figure_kwargs={"figsize": (1000, 200), "figsize_units": "dots"},
            backend=backend,
        )
        assert "plot" in pc.viz.data_vars

    def test_plot_convergence_dist(self, datatree, backend):
        pc = plot_convergence_dist(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "rhat" in pc.viz["dist"]
        assert "ess_bulk" in pc.viz["title"]

    @pytest.mark.parametrize("kind", ["kde", "hist", "ecdf"])
    def test_plot_dist(self, datatree, backend, kind):
        pc = plot_dist(datatree, backend=backend, kind=kind)
        assert not pc.aes
        assert "mu" in pc.viz["dist"].data_vars
        visuals = ("plot", "dist", "credible_interval", "point_estimate")
        assert "face" not in pc.viz.children
        assert "dist" in pc.viz.children
        assert all("hierarchy" not in pc.viz[visual]["mu"].dims for visual in visuals)
        assert all("hierarchy" in pc.viz[visual]["theta"].dims for visual in visuals)

    @pytest.mark.parametrize("kind", ["kde", "hist", "ecdf"])
    def test_plot_dist_filled(self, datatree, backend, kind):
        visuals = {"face": {}}
        pc = plot_dist(datatree, backend=backend, kind=kind, visuals=visuals)
        assert not pc.aes
        assert "mu" in pc.viz["dist"].data_vars
        visuals = ("plot", "dist", "face", "credible_interval", "point_estimate")
        assert "face" in pc.viz.children
        assert "dist" in pc.viz.children
        assert all("hierarchy" not in pc.viz[visual]["mu"].dims for visual in visuals)
        assert all("hierarchy" in pc.viz[visual]["theta"].dims for visual in visuals)

    @pytest.mark.parametrize("kind", ["kde", "hist", "ecdf"])
    def test_plot_dist_sample(self, datatree_sample, backend, kind):
        pc = plot_dist(datatree_sample, backend=backend, sample_dims="sample", kind=kind)
        assert not pc.aes
        assert "mu" in pc.viz["dist"].data_vars
        visuals = ("plot", "dist", "credible_interval", "point_estimate")
        assert "face" not in pc.viz.children
        assert "dist" in pc.viz.children
        assert all("hierarchy" not in pc.viz[visual]["mu"].dims for visual in visuals)
        assert all("hierarchy" in pc.viz[visual]["theta"].dims for visual in visuals)

    @pytest.mark.parametrize("kind", ["kde", "hist", "ecdf"])
    def test_plot_dist_sample_filled(self, datatree_sample, backend, kind):
        visuals = {"face": {}}
        pc = plot_dist(
            datatree_sample, backend=backend, sample_dims="sample", kind=kind, visuals=visuals
        )
        assert not pc.aes
        assert "mu" in pc.viz["dist"].data_vars
        visuals = ("plot", "dist", "face", "credible_interval", "point_estimate")
        assert "face" in pc.viz.children
        assert "dist" in pc.viz.children
        assert all("hierarchy" not in pc.viz[visual]["mu"].dims for visual in visuals)
        assert all("hierarchy" in pc.viz[visual]["theta"].dims for visual in visuals)

    @pytest.mark.parametrize("kind", ["kde"])
    def test_plot_dist_models(self, datatree, datatree2, backend, kind):
        pc = plot_dist({"c": datatree, "n": datatree2}, backend=backend, kind=kind)
        assert "/color" in pc.aes.groups
        assert tuple(pc.aes["color"].dims) == ("model",)
        assert "dist" in pc.viz.children
        assert "mu" in pc.viz["dist"].data_vars
        assert "hierarchy" not in pc.viz["dist"]["mu"].dims
        assert "model" in pc.viz["dist"]["mu"].dims

    def test_plot_ecdf_pit(self, datatree, backend):
        pc = plot_ecdf_pit(datatree, backend=backend, group="prior")
        assert "figure" in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "mu" in pc.viz["ecdf_lines"].data_vars
        assert "hierarchy" not in pc.viz["ecdf_lines"]["mu"].dims
        assert "hierarchy" in pc.viz["ecdf_lines"]["theta"].dims

    def test_plot_energy_dist(self, datatree, backend):
        pc = plot_energy(datatree, backend=backend)
        assert pc is not None
        assert hasattr(pc, "viz")
        assert "/dist" in pc.viz.groups
        assert "energy" in pc.viz["dist"]
        assert "energy" in pc.viz["dist"].coords
        kde_values = pc.viz["dist"]["energy_"].values
        assert kde_values.size > 0
        assert "component_group" not in pc.viz["dist"]["energy_"].dims
        assert "alpha" not in pc.viz["dist"]["energy_"].dims
        energy_coords = pc.viz["dist"]["energy_"].coords["energy"].values
        assert "marginal" in energy_coords
        assert "transition" in energy_coords

    def test_plot_energy_dist_sample(self, datatree_sample, backend):
        pc = plot_energy(datatree_sample, backend=backend)
        assert pc is not None
        assert hasattr(pc, "viz")
        assert "/dist" in pc.viz.groups
        assert "energy" in pc.viz["dist"]
        assert "energy" in pc.viz["dist"].coords
        kde_values = pc.viz["dist"]["energy_"].values
        assert kde_values.size > 0
        assert "component_group" not in pc.viz["dist"]["energy_"].dims
        assert "alpha" not in pc.viz["dist"]["energy_"].dims
        energy_coords = pc.viz["dist"]["energy_"].coords["energy"].values
        assert "marginal" in energy_coords
        assert "transition" in energy_coords

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
        assert "chain" in pc.viz["rug"]["mu"].dims  # checking rug visual overlay
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
            aes={"color": [pseudo_dim]},
            aes_by_visuals={"labels": ["color"]},
            shade_label=pseudo_dim,
            backend=backend,
        )
        assert "plot" in pc.viz.data_vars
        assert "shade" in pc.viz.children
        if pseudo_dim != "__variable__":
            assert pc.aes["alpha"]["neutral_element"].item() == 0
            assert 0 in pc.aes["alpha"]["mapping"].values
            assert pseudo_dim in pc.viz["shade"].dims

    @pytest.mark.parametrize("coverage", (True, False))
    def test_plot_loo_pit(self, datatree, coverage, backend):
        pc = plot_loo_pit(datatree, coverage=coverage, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "y" in pc.viz["plot"].data_vars
        assert "ecdf_lines" in pc.viz.children

    def test_plot_mcse(self, datatree, backend):
        pc = plot_mcse(datatree, backend=backend, rug=True)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "mu" in pc.viz["mcse"]
        assert "mu" in pc.viz["title"]
        assert "mu" in pc.viz["rug"]
        assert "hierarchy" not in pc.viz["mcse"]["mu"].dims
        assert "hierarchy" in pc.viz["mcse"]["theta"].dims
        assert "chain" in pc.viz["rug"]["mu"].dims  # checking rug visual overlay
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

    def test_plot_pair(self, datatree, backend):
        visuals = {"divergence": True}
        pc = plot_pair(
            datatree,
            var_names=["mu", "tau", "theta"],
            coords={"hierarchy": 0},
            triangle="both",
            visuals=visuals,
            backend=backend,
        )
        assert "figure" in pc.viz.data_vars
        assert "divergence" in pc.viz.data_vars
        assert "scatter" in pc.viz.data_vars
        assert "xlabel" in pc.viz.data_vars
        assert "ylabel" in pc.viz.data_vars
        assert "chain" in pc.viz["scatter"].dims
        assert "chain" in pc.viz["divergence"].dims
        assert "chain" not in pc.viz["xlabel"].dims
        assert "chain" not in pc.viz["ylabel"].dims
        assert "col_index" in pc.viz["xlabel"].dims
        assert "row_index" in pc.viz["ylabel"].dims
        assert pc.viz["scatter"].dims == ("row_index", "col_index", "chain")

    @pytest.mark.parametrize("triangle", ("both", "upper", "lower"))
    @pytest.mark.parametrize("marginal", (True, False))
    def test_plot_pair_triangle(self, datatree, marginal, triangle, backend):
        visuals = {"divergence": True}
        pc = plot_pair(
            datatree,
            var_names=["mu", "tau", "theta"],
            coords={"hierarchy": 0},
            marginal=marginal,
            triangle=triangle,
            visuals=visuals,
            backend=backend,
        )
        rows = pc.viz.row_index.values
        cols = pc.viz.col_index.values
        for row_no in rows:
            for col_no in cols:
                if row_no == col_no:
                    assert pc.viz.scatter[row_no, col_no].values[0] is None
                elif row_no > col_no:
                    if triangle in ("lower", "both"):
                        assert pc.viz.scatter[row_no, col_no].values[0] is not None
                    else:
                        assert pc.viz.scatter[row_no, col_no].values[0] is None
                else:
                    if triangle in ("upper", "both"):
                        assert pc.viz.scatter[row_no, col_no].values[0] is not None
                    else:
                        assert pc.viz.scatter[row_no, col_no].values[0] is None

    def test_plot_pair_sample(self, datatree_sample, backend):
        visuals = {"divergence": True}
        sample_dims = ["sample"]
        pc = plot_pair(
            datatree_sample,
            var_names=["mu", "tau", "theta"],
            coords={"hierarchy": 0},
            marginal=True,
            marginal_kind="kde",
            triangle="both",
            sample_dims=sample_dims,
            visuals=visuals,
            backend=backend,
        )
        assert "figure" in pc.viz.data_vars
        assert "divergence" in pc.viz.data_vars
        assert "scatter" in pc.viz.data_vars
        assert "xlabel" in pc.viz.data_vars
        assert "ylabel" in pc.viz.data_vars
        assert "chain" not in pc.viz["scatter"].dims
        assert "chain" not in pc.viz["divergence"].dims
        assert "chain" not in pc.viz["xlabel"].dims
        assert "chain" not in pc.viz["ylabel"].dims
        assert "col_index" in pc.viz["xlabel"].dims
        assert "row_index" in pc.viz["ylabel"].dims
        assert pc.viz["divergence"].dims == ("row_index", "col_index")
        assert pc.viz["scatter"].dims == ("row_index", "col_index")

    def test_plot_pair_focus(self, datatree, backend):
        visuals = {"divergence": True}
        pc = plot_pair_focus(
            datatree,
            focus_var=datatree.posterior["theta"].sel(hierarchy=0),
            backend=backend,
            visuals=visuals,
        )
        assert "figure" in pc.viz.data_vars
        assert "scatter" in pc.viz.children
        assert "divergence" in pc.viz.children
        assert "chain" in pc.viz["scatter"]["mu"].dims
        assert "chain" in pc.viz["divergence"]["mu"].dims
        assert "hierarchy" not in pc.viz["scatter"]["mu"].dims
        assert "hierarchy" in pc.viz["scatter"]["theta"].dims
        assert pc.viz["xlabel"]["mu"].dims == ()
        assert pc.viz["xlabel"]["theta"].dims == ("hierarchy",)

    def test_plot_pair_focus_sample(self, datatree_sample, backend):
        visuals = {"divergence": True}
        sample_dims = ["sample"]
        pc = plot_pair_focus(
            datatree_sample,
            focus_var=datatree_sample.posterior["theta"].sel(hierarchy=0),
            sample_dims=sample_dims,
            backend=backend,
            visuals=visuals,
        )
        assert "figure" in pc.viz.data_vars
        assert "scatter" in pc.viz.children
        assert "divergence" in pc.viz.children
        assert "chain" not in pc.viz["scatter"]["mu"].dims
        assert "chain" not in pc.viz["divergence"]["mu"].dims
        assert "hierarchy" not in pc.viz["scatter"]["mu"].dims
        assert "hierarchy" in pc.viz["scatter"]["theta"].dims
        assert pc.viz["xlabel"]["mu"].dims == ()
        assert pc.viz["xlabel"]["theta"].dims == ("hierarchy",)

    @pytest.mark.parametrize("norm_method", (None, "normal", "minmax", "rank"))
    def test_plot_parallel(self, datatree, norm_method, backend):
        visuals = {"xticks": {"rotation": 30}}
        pc = plot_parallel(
            datatree,
            var_names=["theta", "tau", "mu"],
            norm_method=norm_method,
            label_type="vert",
            visuals=visuals,
            backend=backend,
        )
        assert "figure" in pc.viz.data_vars
        assert "line" in pc.viz.children
        assert "diverging" in pc.viz["line"].dims
        assert "xticks" in pc.viz.children
        assert "diverging" not in pc.viz["xticks"].dims
        assert "labels" in pc.viz["xticks"].dims

    @pytest.mark.parametrize("norm_method", (None, "normal", "minmax", "rank"))
    def test_plot_parallel_sample(self, datatree_sample, norm_method, backend):
        visuals = {"xticks": {"rotation": 30}}
        pc = plot_parallel(
            datatree_sample,
            var_names=["theta", "tau", "mu"],
            norm_method=norm_method,
            label_type="vert",
            visuals=visuals,
            sample_dims="sample",
            backend=backend,
        )
        assert "figure" in pc.viz.data_vars
        assert "line" in pc.viz.children
        assert "diverging" in pc.viz["line"].dims
        assert "xticks" in pc.viz.children
        assert "diverging" not in pc.viz["xticks"].dims
        assert "labels" in pc.viz["xticks"].dims

    @pytest.mark.parametrize("kind", ["kde", "ecdf", "hist"])
    def test_plot_ppc_dist(self, datatree, kind, backend):
        pc = plot_ppc_dist(datatree, kind=kind, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "/overlay_ppc" in pc.aes.groups
        assert "y" in pc.viz["predictive_dist"]
        assert "y" in pc.viz["observed_dist"]

    def test_plot_ppc_pava(self, datatree_binary, backend):
        pc = plot_ppc_pava(datatree_binary, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "lines" in pc.viz.children
        assert "y" in pc.viz["plot"]

    @pytest.mark.parametrize("coverage", [False, True])
    def test_plot_ppc_pit(self, datatree, coverage, backend):
        pc = plot_ppc_pit(datatree, coverage=coverage, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "y" in pc.viz["plot"]
        assert "ecdf_lines" in pc.viz.children

    def test_plot_ppc_rootogram(self, datatree3, backend):
        pc = plot_ppc_rootogram(datatree3, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "predictive_markers" in pc.viz.children
        assert "y" in pc.viz["plot"]

    def test_plot_ppc_rootogram_continuous_error(self, datatree, backend):
        with pytest.raises(ValueError, match="Detected at least one continuous variable"):
            plot_ppc_rootogram(datatree, backend=backend)

    @pytest.mark.parametrize("kind", ["kde", "ecdf", "hist"])
    def test_plot_ppc_tstat(self, datatree, kind, backend):
        pc = plot_ppc_tstat(datatree, kind=kind, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "dist" in pc.viz.children
        assert "observed_tstat" in pc.viz.children
        assert "y" in pc.viz["plot"]

    def test_plot_prior_posterior(self, datatree, backend):
        pc = plot_prior_posterior(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "group" not in pc.viz["plot"].coords
        assert "group" in pc.viz["dist"].coords

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

    def test_plot_psense_quantities(self, datatree, backend):
        pc = plot_psense_quantities(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "quantities" in pc.viz["plot"]["mu"].dims
        assert "mean" in pc.viz["quantities"]

    def test_plot_rank(self, datatree, backend):
        pc = plot_rank(datatree, backend=backend)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "hierarchy" not in pc.viz["plot"]["mu"].dims
        assert "hierarchy" in pc.viz["plot"]["theta"].dims
        assert "/color" in pc.aes.groups
        assert "mapping" in pc.aes["overlay"]
        assert "chain" in pc.aes["overlay"]["mapping"].dims

    @pytest.mark.parametrize("compact", (True, False))
    @pytest.mark.parametrize("combined", (True, False))
    def test_plot_rank_dist(self, datatree, backend, compact, combined):
        pc = plot_rank_dist(datatree, backend=backend, compact=compact, combined=combined)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        if combined:
            assert "chain" not in pc.viz["dist"]["theta"].dims
        else:
            assert "chain" in pc.viz["dist"]["theta"].dims
        if compact:
            assert "hierarchy" not in pc.viz["plot"]["theta"].dims
        else:
            assert "hierarchy" in pc.viz["plot"]["theta"].dims

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
            aes={"color": [pseudo_dim]},
            aes_by_visuals={"labels": ["color"]},
            shade_label=pseudo_dim,
            backend=backend,
        )
        assert "plot" in pc.viz.data_vars
        assert "shade" in pc.viz.children
        if pseudo_dim != "__variable__":
            assert pc.aes["alpha"]["neutral_element"].item() == 0
            assert 0 in pc.aes["alpha"]["mapping"].values
            assert pseudo_dim in pc.viz["shade"].dims

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
        pc = plot_trace_dist(datatree, backend=backend, compact=compact, combined=combined)
        assert "figure" in pc.viz.data_vars
        assert "plot" not in pc.viz.data_vars
        assert "plot" in pc.viz.children
        assert "chain" in pc.viz["trace"]["theta"].dims
        if combined:
            assert "chain" not in pc.viz["dist"]["theta"].dims
        else:
            assert "chain" in pc.viz["dist"]["theta"].dims
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
        add_lines(pc, [0, 1], aes_by_visuals={"ref_line": ["color"]})
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
        add_bands(pc, [(0, 1), (2, 5)], aes_by_visuals={"ref_band": ["color"]})
        assert "mu" in pc.viz["ref_band"].data_vars
        assert "ref_dim" in pc.viz["ref_band"]["mu"].dims
        assert "/color" in pc.aes.groups
        assert "ref_dim" in pc.aes["color"].dims
