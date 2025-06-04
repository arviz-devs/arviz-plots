# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots using the none backend."""
import arviz_stats  # pylint: disable=unused-import
import hypothesis.strategies as st
import numpy as np
import pytest
from arviz_base import from_dict
from hypothesis import given
from scipy.stats import halfnorm, norm

from arviz_plots import (
    plot_convergence_dist,
    plot_dist,
    plot_ess,
    plot_ess_evolution,
    plot_forest,
    plot_psense_dist,
    plot_rank_dist,
    plot_ridge,
)

pytestmark = pytest.mark.usefixtures("no_artist_kwargs")


@pytest.fixture(scope="module")
def datatree(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(3, 50))
    tau = np.exp(rng.normal(size=(3, 50, 2)))
    theta = rng.normal(size=(3, 50, 2, 3))
    mu_prior = norm(0, 3).logpdf(mu)
    tau_prior = halfnorm(scale=5).logpdf(tau)
    theta_prior = norm(0, 1).logpdf(theta)
    theta_orig = rng.uniform(size=3)
    idxs0 = rng.choice(np.arange(2), size=29)
    idxs1 = rng.choice(np.arange(3), size=29)
    x = np.linspace(0, 1, 29)
    obs = rng.normal(loc=x + theta_orig[idxs1], scale=3)
    log_lik = norm(
        mu[:, :, None] * x[None, None, :] + theta[:, :, idxs0, idxs1], tau[:, :, idxs0]
    ).logpdf(obs[None, None, :])
    log_lik = log_lik / log_lik.var()
    diverging = rng.choice([True, False], size=(3, 50), p=[0.1, 0.9])

    dt = from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau},
            "log_prior": {"mu": mu_prior, "theta": theta_prior, "tau": tau_prior},
            "log_likelihood": {"y": log_lik},
            "observed_data": {"y": obs},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy", "group"], "tau": ["hierarchy"]},
    )
    dt["point_estimate"] = dt.posterior.mean(("chain", "draw"))
    # TODO: should become dt.azstats.eti() after fix in arviz-stats
    post = dt.posterior.ds
    dt["trunk"] = post.azstats.eti(prob=0.5)
    dt["twig"] = post.azstats.eti(prob=0.9)
    return dt


kind_value = st.sampled_from(("kde", "ecdf"))
ess_kind_value = st.sampled_from(("local", "quantile"))
ci_kind_value = st.sampled_from(("eti", "hdi"))
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
    for artist, value in visuals.items():
        if value is False:
            assert artist not in pc.viz.children
        else:
            assert artist in pc.viz.children
            assert all(
                var_name in pc.viz[artist].data_vars for var_name in datatree["posterior"].data_vars
            )


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
    labels_shade_label=labels_shade(st.sampled_from(("__variable__", "hierarchy", "group"))),
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
    for artist, value in visuals.items():
        if value is False:
            assert artist not in pc.viz.children
        elif artist == "labels":
            assert all(f"{label.strip('_')}_label" in pc.viz.children for label in labels)
        elif artist == "shade":
            if shade_label is None:
                assert artist not in pc.viz.children
            else:
                assert artist in pc.viz.children
        elif artist != "ticklabels":
            assert artist in pc.viz.children
            assert all(
                var_name in pc.viz[artist].data_vars for var_name in datatree["posterior"].data_vars
            )


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
    labels_shade_label=labels_shade(st.sampled_from(("__variable__", "hierarchy", "group"))),
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
    for artist, value in visuals.items():
        if value is False:
            assert artist not in pc.viz.children
        elif artist == "labels":
            assert all(f"{label.strip('_')}_label" in pc.viz.children for label in labels)
        elif artist == "shade":
            if shade_label is None:
                assert artist not in pc.viz.children
            else:
                assert artist in pc.viz.children
        elif artist != "ticklabels":
            assert artist in pc.viz.children
            assert all(
                var_name in pc.viz[artist].data_vars for var_name in datatree["posterior"].data_vars
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
    for artist, value in visuals.items():
        if value is False:
            assert artist not in pc.viz.children
        elif artist in ["mean", "sd", "mean_text", "sd_text"]:
            if extra_methods is False:
                assert artist not in pc.viz.children
            else:
                assert artist in pc.viz.children
        elif artist == "rug":
            if rug is False:
                assert artist not in pc.viz.children
            else:
                assert artist in pc.viz.children
        else:
            assert artist in pc.viz.children


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
    for artist, value in visuals.items():
        if value is False:
            assert artist not in pc.viz.children
        elif artist in ["mean", "sd", "mean_text", "sd_text"]:
            if extra_methods is False:
                assert artist not in pc.viz.children
            else:
                assert artist in pc.viz.children
        else:
            assert artist in pc.viz.children


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
    for artist, value in visuals.items():
        if value is False:
            assert artist not in pc.viz.children
        else:
            assert artist in pc.viz.children


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
    for artist, value in visuals.items():
        if value is False:
            assert artist not in pc.viz.children
        elif artist == "ref_line":
            if ref_line:
                assert artist in pc.viz.children
            else:
                assert artist not in pc.viz.children
        else:
            assert artist in pc.viz.children


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
    for artist, value in visuals.items():
        if value is False:
            assert artist not in pc.viz.children
        else:
            assert artist in pc.viz.children
