# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots using the none backend."""
import arviz_stats  # pylint: disable=unused-import
import hypothesis.strategies as st
import pytest
from hypothesis import given

from arviz_plots import (
    plot_autocorr,
    plot_bf,
    plot_convergence_dist,
    plot_dist,
    plot_ecdf_pit,
    plot_ess,
    plot_ess_evolution,
    plot_forest,
    plot_loo_pit,
    plot_mcse,
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
)

pytestmark = pytest.mark.usefixtures("no_artist_kwargs")

kind_value = st.sampled_from(("kde", "ecdf"))
ess_kind_value = st.sampled_from(("local", "quantile"))
t_stat_value = st.sampled_from(("mean", "median", "std", "var", "min", "max", "iqr", "0.5", 0.5))
ci_kind_value = st.sampled_from(("eti", "hdi"))
ci_prob_value = st.floats(min_value=0.1, max_value=0.99, allow_nan=False, allow_infinity=False)
point_estimate_value = st.sampled_from(("mean", "median"))
visuals_value = st.sampled_from(({}, False, {"color": "red"}))
visuals_value_no_false = st.sampled_from(({}, {"color": "red"}))


@st.composite
def labels_shade(draw, elements):
    labels = draw(st.lists(elements, unique=True))
    i = draw(st.integers(min_value=-1, max_value=len(labels) - 1))
    if i == -1:
        return (labels, None)
    return (labels, labels[i])


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "lines": visuals_value,
            "ref_line": visuals_value,
            "ci": visuals_value,
            "xlabel": visuals_value,
            "title": visuals_value,
        },
    ),
)
def test_plot_autocorr(datatree, visuals):
    pc = plot_autocorr(
        datatree,
        backend="none",
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "kind": visuals_value_no_false,
            "ref_line": visuals_value_no_false,
            "title": visuals_value,
        },
    ),
    kind=kind_value,
    ref_val=st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_plot_bf(datatree, kind, ref_val, visuals):
    kind_kwargs = visuals.pop("kind", None)
    if kind_kwargs is not None:
        visuals[kind] = kind_kwargs
    pc = plot_bf(
        datatree,
        backend="none",
        var_names="mu",
        kind=kind,
        ref_val=ref_val,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "kind": visuals_value,
            "ref_line": visuals_value_no_false,
            "title": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    diagnostics=st.sampled_from(
        [
            # fmt: off
            None, "rhat", "rhat_rank", "rhat_folded", "rhat_z_scale", "rhat_split",
            "rhat_identity", "ess_bulk", "ess_tail", "ess_mean", "ess_sd",
            "ess_quantile(0.9)", "ess_local(0.1, 0.9)", "ess_median", "ess_mad",
            "ess_z_scale", "ess_folded", "ess_identity"
            # fmt: on
        ]
    ),
    kind=kind_value,
    ref_line=st.booleans(),
)
def test_plot_convergence_dist(datatree, diagnostics, kind, ref_line, visuals):
    kind_kwargs = visuals.pop("kind", None)
    if kind_kwargs is not None:
        visuals[kind] = kind_kwargs
    pc = plot_convergence_dist(
        datatree,
        diagnostics=diagnostics,
        backend="none",
        kind=kind,
        ref_line=ref_line,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    if diagnostics is None:
        diagnostics = ["ess_bulk", "ess_tail", "rhat"]
    if isinstance(diagnostics, str):
        diagnostics = [diagnostics]
    assert all(
        diagnostic in child.data_vars
        for diagnostic in diagnostics
        for child in pc.viz.children.values()
    )
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        elif visual == "ref_line":
            if ref_line:
                assert visual in pc.viz.children
            else:
                assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "kind": visuals_value,
            "credible_interval": visuals_value,
            "point_estimate": visuals_value,
            "point_estimate_text": visuals_value,
            "title": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    kind=kind_value,
    ci_kind=ci_kind_value,
    point_estimate=point_estimate_value,
)
def test_plot_dist(datatree, kind, ci_kind, point_estimate, visuals):
    kind_kwargs = visuals.pop("kind", None)
    if kind_kwargs is not None:
        visuals[kind] = kind_kwargs
    pc = plot_dist(
        datatree,
        backend="none",
        kind=kind,
        ci_kind=ci_kind,
        point_estimate=point_estimate,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "ecdf_lines": visuals_value,
            "ci": visuals_value,
            "xlabel": visuals_value,
            "ylabel": visuals_value,
            "title": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    ci_prob=ci_prob_value,
)
def test_plot_ecdf_pit(datatree, ci_prob, visuals):
    pc = plot_ecdf_pit(
        datatree,
        group="prior",
        backend="none",
        ci_prob=ci_prob,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["prior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "ess": visuals_value,
            "rug": visuals_value_no_false,
            "xlabel": visuals_value_no_false,
            "ylabel": visuals_value_no_false,
            "mean": visuals_value,
            "mean_text": visuals_value,
            "sd": visuals_value,
            "sd_text": visuals_value,
            "min_ess": visuals_value,
            "title": visuals_value,
        },
    ),
    kind=ess_kind_value,
    relative=st.booleans(),
    rug=st.booleans(),
    n_points=st.integers(min_value=1, max_value=5),
    extra_methods=st.booleans(),
    min_ess=st.integers(min_value=10, max_value=150),
)
def test_plot_ess(datatree, kind, relative, rug, n_points, extra_methods, min_ess, visuals):
    pc = plot_ess(
        datatree,
        backend="none",
        kind=kind,
        relative=relative,
        rug=rug,
        n_points=n_points,
        extra_methods=extra_methods,
        min_ess=min_ess,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        elif visual in ["mean", "sd", "mean_text", "sd_text"]:
            if extra_methods is False:
                assert visual not in pc.viz.children
            else:
                assert visual in pc.viz.children
        elif visual == "rug":
            if rug is False:
                assert visual not in pc.viz.children
            else:
                assert visual in pc.viz.children
        else:
            assert visual in pc.viz.children


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "ess_bulk": visuals_value,
            "ess_bulk_line": visuals_value,
            "ess_tail": visuals_value,
            "ess_tail_line": visuals_value,
            "xlabel": visuals_value_no_false,
            "ylabel": visuals_value_no_false,
            "mean": visuals_value,
            "mean_text": visuals_value,
            "sd": visuals_value,
            "sd_text": visuals_value,
            "min_ess": visuals_value,
            "title": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    relative=st.booleans(),
    min_ess=st.integers(min_value=10, max_value=150),
    extra_methods=st.booleans(),
    n_points=st.integers(min_value=2, max_value=12),
)
def test_plot_ess_evolution(datatree, relative, n_points, extra_methods, min_ess, visuals):
    pc = plot_ess_evolution(
        datatree,
        backend="none",
        relative=relative,
        n_points=n_points,
        extra_methods=extra_methods,
        min_ess=min_ess,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        elif visual in ["mean", "sd", "mean_text", "sd_text"]:
            if extra_methods is False:
                assert visual not in pc.viz.children
            else:
                assert visual in pc.viz.children
        else:
            assert visual in pc.viz.children


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "trunk": visuals_value,
            "twig": visuals_value,
            "point_estimate": visuals_value,
            "labels": st.sampled_from(({}, {"color": "red"})),
            "shade": st.sampled_from(({}, {"color": "red"})),
            "ticklabels": st.sampled_from(({}, False)),
            "remove_axis": st.just(False),
        },
    ),
    stats=st.fixed_dictionaries(
        {},
        optional={
            "trunk": st.just(True),
            "twig": st.just(True),
            "point_estimate": st.just(True),
        },
    ),
    combined=st.booleans(),
    ci_kind=ci_kind_value,
    point_estimate=point_estimate_value,
    labels_shade_label=labels_shade(st.sampled_from(("__variable__", "hierarchy"))),
)
def test_plot_forest(
    datatree, combined, ci_kind, point_estimate, visuals, stats, labels_shade_label
):
    labels = labels_shade_label[0]
    shade_label = labels_shade_label[1]
    stats = {key: datatree[key].ds for key in stats}
    pc = plot_forest(
        datatree,
        backend="none",
        combined=combined,
        ci_kind=ci_kind,
        point_estimate=point_estimate,
        labels=labels,
        shade_label=shade_label,
        visuals=visuals,
        stats=stats,
    )
    assert "plot" in pc.viz.data_vars
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        elif visual == "labels":
            assert all(f"{label.strip('_')}_label" in pc.viz.children for label in labels)
        elif visual == "shade":
            if shade_label is None:
                assert visual not in pc.viz.children
            else:
                assert visual in pc.viz.children
        elif visual != "ticklabels":
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "ecdf_lines": visuals_value,
            "ci": visuals_value,
            "xlabel": visuals_value,
            "ylabel": visuals_value,
            "title": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    ci_prob=ci_prob_value,
    coverage=st.booleans(),
)
def test_plot_loo_pit(datatree, ci_prob, coverage, visuals):
    pc = plot_loo_pit(
        datatree,
        backend="none",
        ci_prob=ci_prob,
        coverage=coverage,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars
                for var_name in datatree["posterior_predictive"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "mcse": visuals_value,
            "rug": visuals_value_no_false,
            "xlabel": visuals_value_no_false,
            "ylabel": visuals_value_no_false,
            "mean": visuals_value,
            "mean_text": visuals_value,
            "sd": visuals_value,
            "sd_text": visuals_value,
            "title": visuals_value,
        },
    ),
    rug=st.booleans(),
    n_points=st.integers(min_value=1, max_value=5),
    extra_methods=st.booleans(),
)
def test_plot_mcse(datatree, rug, n_points, extra_methods, visuals):
    pc = plot_mcse(
        datatree,
        backend="none",
        rug=rug,
        n_points=n_points,
        extra_methods=extra_methods,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        elif visual in ["mean", "sd", "mean_text", "sd_text"]:
            if extra_methods is False:
                assert visual not in pc.viz.children
            else:
                assert visual in pc.viz.children
        elif visual == "rug":
            if rug is False:
                assert visual not in pc.viz.children
            else:
                assert visual in pc.viz.children
        else:
            assert visual in pc.viz.children


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "kind": visuals_value_no_false,
            "title": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    kind=kind_value,
)
def test_plot_ppc_dist(datatree, kind, visuals):
    kind_kwargs = visuals.pop("kind", None)
    if kind_kwargs is not None:
        visuals[kind] = kind_kwargs
    pc = plot_ppc_dist(
        datatree,
        backend="none",
        kind=kind,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars
                for var_name in datatree["posterior_predictive"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "lines": visuals_value,
            "markers": visuals_value,
            "reference_line": visuals_value,
            "ci": visuals_value,
            "xlabel": visuals_value,
            "ylabel": visuals_value,
            "title": visuals_value,
        },
    ),
    ci_prob=ci_prob_value,
)
def test_plot_ppc_pava(datatree_binary, ci_prob, visuals):
    pc = plot_ppc_pava(
        datatree_binary,
        backend="none",
        ci_prob=ci_prob,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars
                for var_name in datatree_binary["posterior_predictive"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "predictive_markers": visuals_value,
            "observed_markers": visuals_value,
            "ci": visuals_value,
            "xlabel": visuals_value,
            "ylabel": visuals_value,
            "grid": visuals_value,
            "title": visuals_value,
        },
    ),
    ci_prob=ci_prob_value,
)
def test_plot_ppc_rootogram(datatree3, ci_prob, visuals):
    pc = plot_ppc_rootogram(
        datatree3,
        backend="none",
        ci_prob=ci_prob,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars
                for var_name in datatree3["posterior_predictive"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "ecdf_lines": visuals_value,
            "ci": visuals_value,
            "xlabel": visuals_value,
            "ylabel": visuals_value,
            "title": visuals_value,
        },
    ),
    coverage=st.booleans(),
    ci_prob=ci_prob_value,
)
def test_plot_ppc_pit(datatree, coverage, ci_prob, visuals):
    pc = plot_ppc_pit(
        datatree,
        backend="none",
        coverage=coverage,
        ci_prob=ci_prob,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars
                for var_name in datatree["posterior_predictive"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "kind": visuals_value_no_false,
            "observed_tstat": visuals_value,
            "credible_interval": visuals_value,
            "point_estimate": visuals_value,
            "point_estimate_text": visuals_value,
            "title": visuals_value,
            "rug": st.booleans(),
            "remove_axis": st.just(False),
        },
    ),
    kind=kind_value,
    t_stat=t_stat_value,
)
def test_plot_ppc_tstat(datatree, kind, t_stat, visuals):
    if kind != "kde":
        visuals.pop("rug", None)
    kind_kwargs = visuals.pop("kind", None)
    if kind_kwargs is not None:
        visuals[kind] = kind_kwargs
    pc = plot_ppc_tstat(
        datatree,
        backend="none",
        kind=kind,
        t_stat=t_stat,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars
                for var_name in datatree["posterior_predictive"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "kind": visuals_value_no_false,
            "title": visuals_value,
        },
    ),
    kind=kind_value,
)
def test_plot_prior_posterior(datatree, kind, visuals):
    kind_kwargs = visuals.pop("kind", None)
    if kind_kwargs is not None:
        visuals[kind] = kind_kwargs
    pc = plot_prior_posterior(
        datatree,
        backend="none",
        kind=kind,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "kind": visuals_value,
            "credible_interval": visuals_value,
            "point_estimate": visuals_value,
            "point_estimate_text": visuals_value,
            "title": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    alphas=st.sampled_from(((0.9, 1.1), None)),
    kind=kind_value,
    point_estimate=point_estimate_value,
    ci_kind=ci_kind_value,
)
def test_plot_psense(datatree, alphas, kind, point_estimate, ci_kind, visuals):
    kind_kwargs = visuals.pop("kind", None)
    if kind_kwargs is not None:
        visuals[kind] = kind_kwargs
    pc = plot_psense_dist(
        datatree,
        alphas=alphas,
        backend="none",
        kind=kind,
        ci_kind=ci_kind,
        point_estimate=point_estimate,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "prior_markers": visuals_value,
            "prior_lines": visuals_value,
            "likelihood_markers": visuals_value,
            "likelihood_lines": visuals_value,
            "mcse": visuals_value,
            # "ticks": visuals_value,
            "title": visuals_value,
        },
    ),
    quantities=st.sampled_from(
        (
            "mean",
            "sd",
            "median",
        )
    ),
    mcse=st.booleans(),
)
def test_plot_psense_quantities(datatree, quantities, mcse, visuals):
    pc = plot_psense_quantities(
        datatree,
        backend="none",
        quantities=quantities,
        mcse=mcse,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        elif mcse is False and visual == "mcse":
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "ecdf_lines": visuals_value,
            "ci": visuals_value,
            "xlabel": visuals_value,
            "title": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    ci_prob=ci_prob_value,
)
def test_plot_rank(datatree, ci_prob, visuals):
    pc = plot_rank(
        datatree,
        backend="none",
        ci_prob=ci_prob,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "kind": visuals_value,
        },
    ),
    kind=kind_value,
    compact=st.booleans(),
    combined=st.booleans(),
)
def test_plot_rank_dist(datatree, kind, compact, combined, visuals):
    kind_kwargs = visuals.pop("kind", None)
    if kind_kwargs is not None:
        visuals[kind] = kind_kwargs
    pc = plot_rank_dist(
        datatree,
        backend="none",
        kind=kind,
        compact=compact,
        combined=combined,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "edge": visuals_value,
            "face": visuals_value,
            "labels": st.sampled_from(({}, {"color": "red"})),
            "shade": st.sampled_from(({}, {"color": "red"})),
            "ticklabels": st.sampled_from(({}, False)),
            "remove_axis": st.just(False),
        },
    ),
    combined=st.booleans(),
    labels_shade_label=labels_shade(st.sampled_from(("__variable__", "hierarchy"))),
)
def test_plot_ridge(datatree, combined, visuals, labels_shade_label):
    labels = labels_shade_label[0]
    shade_label = labels_shade_label[1]
    pc = plot_ridge(
        datatree,
        backend="none",
        combined=combined,
        labels=labels,
        shade_label=shade_label,
        visuals=visuals,
    )
    assert "plot" in pc.viz.data_vars
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        elif visual == "labels":
            assert all(f"{label.strip('_')}_label" in pc.viz.children for label in labels)
        elif visual == "shade":
            if shade_label is None:
                assert visual not in pc.viz.children
            else:
                assert visual in pc.viz.children
        elif visual != "ticklabels":
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "trace": visuals_value,
            "divergence": visuals_value,
            "xlabel": visuals_value,
            "ticklabels": visuals_value,
            "title": visuals_value,
        },
    ),
)
def test_plot_trace(datatree, visuals):
    pc = plot_trace(
        datatree,
        backend="none",
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )


@given(
    visuals=st.fixed_dictionaries(
        {},
        optional={
            "trace": visuals_value,
            # "divergence": visuals_value,
            # "label": visuals_value,
            "ticklabels": visuals_value,
            "xlabel_trace": visuals_value,
            "remove_axis": st.just(False),
        },
    ),
    compact=st.booleans(),
    combined=st.booleans(),
)
def test_plot_trace_dist(datatree, compact, combined, visuals):
    pc = plot_trace_dist(
        datatree,
        backend="none",
        compact=compact,
        combined=combined,
        visuals=visuals,
    )
    assert "plot" in pc.viz.children
    for visual, value in visuals.items():
        if value is False:
            assert visual not in pc.viz.children
        else:
            assert visual in pc.viz.children
            assert all(
                var_name in pc.viz[visual].data_vars for var_name in datatree["posterior"].data_vars
            )
