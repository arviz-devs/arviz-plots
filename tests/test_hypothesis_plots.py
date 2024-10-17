# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots using the none backend."""
import arviz_stats  # pylint: disable=unused-import
import hypothesis.strategies as st
import numpy as np
import pytest
from arviz_base import from_dict
from datatree import DataTree
from hypothesis import given

from arviz_plots import plot_dist, plot_ess, plot_ess_evolution, plot_forest, plot_ridge

pytestmark = pytest.mark.usefixtures("no_artist_kwargs")


@pytest.fixture(scope="module")
def datatree(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(3, 50))
    tau = rng.normal(size=(3, 50, 2))
    theta = rng.normal(size=(3, 50, 2, 3))
    diverging = rng.choice([True, False], size=(3, 50), p=[0.1, 0.9])

    dt = from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy", "group"], "tau": ["hierarchy"]},
    )
    dt["point_estimate"] = dt.posterior.mean(("chain", "draw"))
    # TODO: should become dt.azstats.eti() after fix in arviz-stats
    post = dt.posterior.ds
    DataTree(name="trunk", parent=dt, data=post.azstats.eti(prob=0.5))
    DataTree(name="twig", parent=dt, data=post.azstats.eti(prob=0.9))
    return dt


kind_value = st.sampled_from(("kde", "ecdf"))
ess_kind_value = st.sampled_from(("local", "quantile"))
ci_kind_value = st.sampled_from(("eti", "hdi"))
point_estimate_value = st.sampled_from(("mean", "median"))
plot_kwargs_value = st.sampled_from(({}, False, {"color": "red"}))
plot_kwargs_value_no_false = st.sampled_from(({}, {"color": "red"}))


@st.composite
def labels_shade(draw, elements):
    labels = draw(st.lists(elements, unique=True))
    i = draw(st.integers(min_value=-1, max_value=len(labels) - 1))
    if i == -1:
        return (labels, None)
    return (labels, labels[i])


@given(
    plot_kwargs=st.fixed_dictionaries(
        {},
        optional={
            "kind": plot_kwargs_value,
            "credible_interval": plot_kwargs_value,
            "point_estimate": plot_kwargs_value,
            "point_estimate_text": plot_kwargs_value,
            "title": plot_kwargs_value,
            "remove_axis": st.just(False),
        },
    ),
    kind=kind_value,
    ci_kind=ci_kind_value,
    point_estimate=point_estimate_value,
)
def test_plot_dist(datatree, kind, ci_kind, point_estimate, plot_kwargs):
    kind_kwargs = plot_kwargs.pop("kind", None)
    if kind_kwargs is not None:
        plot_kwargs[kind] = kind_kwargs
    pc = plot_dist(
        datatree,
        backend="none",
        kind=kind,
        ci_kind=ci_kind,
        point_estimate=point_estimate,
        plot_kwargs=plot_kwargs,
    )
    assert all("plot" in child for child in pc.viz.children.values())
    for key, value in plot_kwargs.items():
        if value is False:
            assert all(key not in child for child in pc.viz.children.values())
        elif key != "remove_axis":
            assert all(key in child for child in pc.viz.children.values())


@given(
    plot_kwargs=st.fixed_dictionaries(
        {},
        optional={
            "trunk": plot_kwargs_value,
            "twig": plot_kwargs_value,
            "point_estimate": plot_kwargs_value,
            "labels": st.sampled_from(({}, {"color": "red"})),
            "shade": st.sampled_from(({}, {"color": "red"})),
            "ticklabels": st.sampled_from(({}, False)),
            "remove_axis": st.just(False),
        },
    ),
    stats_kwargs=st.fixed_dictionaries(
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
    datatree, combined, ci_kind, point_estimate, plot_kwargs, stats_kwargs, labels_shade_label
):
    labels = labels_shade_label[0]
    shade_label = labels_shade_label[1]
    stats_kwargs = {key: datatree[key].ds for key in stats_kwargs}
    pc = plot_forest(
        datatree,
        backend="none",
        combined=combined,
        ci_kind=ci_kind,
        point_estimate=point_estimate,
        labels=labels,
        shade_label=shade_label,
        plot_kwargs=plot_kwargs,
        stats_kwargs=stats_kwargs,
    )
    assert all("plot" not in child for child in pc.viz.children.values())
    assert "plot" in pc.viz.data_vars
    for key, value in plot_kwargs.items():
        if value is False:
            assert all(key not in child for child in pc.viz.children.values())
        elif key == "labels":
            for label in labels:
                assert all(
                    f"{label.strip('_')}_label" in child for child in pc.viz.children.values()
                )
        elif key == "shade":
            if shade_label is None:
                assert all(key not in child for child in pc.viz.children.values())
            else:
                assert all(key in child for child in pc.viz.children.values())
        elif key not in ("remove_axis", "ticklabels"):
            assert all(key in child for child in pc.viz.children.values())


@given(
    plot_kwargs=st.fixed_dictionaries(
        {},
        optional={
            "edge": plot_kwargs_value,
            "face": plot_kwargs_value,
            "labels": st.sampled_from(({}, {"color": "red"})),
            "shade": st.sampled_from(({}, {"color": "red"})),
            "ticklabels": st.sampled_from(({}, False)),
            "remove_axis": st.just(False),
        },
    ),
    combined=st.booleans(),
    labels_shade_label=labels_shade(st.sampled_from(("__variable__", "hierarchy", "group"))),
)
def test_plot_ridge(datatree, combined, plot_kwargs, labels_shade_label):
    labels = labels_shade_label[0]
    shade_label = labels_shade_label[1]
    pc = plot_ridge(
        datatree,
        backend="none",
        combined=combined,
        labels=labels,
        shade_label=shade_label,
        plot_kwargs=plot_kwargs,
    )
    assert all("plot" not in child for child in pc.viz.children.values())
    assert "plot" in pc.viz.data_vars
    for key, value in plot_kwargs.items():
        if value is False:
            assert all(key not in child for child in pc.viz.children.values())
        elif key == "labels":
            for label in labels:
                assert all(
                    f"{label.strip('_')}_label" in child for child in pc.viz.children.values()
                )
        elif key == "shade":
            if shade_label is None:
                assert all(key not in child for child in pc.viz.children.values())
            else:
                assert all(key in child for child in pc.viz.children.values())
        elif key not in ("remove_axis", "ticklabels"):
            assert all(key in child for child in pc.viz.children.values())


@given(
    plot_kwargs=st.fixed_dictionaries(
        {},
        optional={
            "ess": plot_kwargs_value,
            "rug": plot_kwargs_value_no_false,
            "xlabel": plot_kwargs_value_no_false,
            "ylabel": plot_kwargs_value_no_false,
            "mean": plot_kwargs_value,
            "mean_text": plot_kwargs_value,
            "sd": plot_kwargs_value,
            "sd_text": plot_kwargs_value,
            "min_ess": plot_kwargs_value,
            "title": plot_kwargs_value,
        },
    ),
    kind=ess_kind_value,
    relative=st.booleans(),
    rug=st.booleans(),
    n_points=st.integers(min_value=1, max_value=5),
    extra_methods=st.booleans(),
    min_ess=st.integers(min_value=10, max_value=150),
)
def test_plot_ess(datatree, kind, relative, rug, n_points, extra_methods, min_ess, plot_kwargs):
    pc = plot_ess(
        datatree,
        backend="none",
        kind=kind,
        relative=relative,
        rug=rug,
        n_points=n_points,
        extra_methods=extra_methods,
        min_ess=min_ess,
        plot_kwargs=plot_kwargs,
    )
    assert all("plot" in child for child in pc.viz.children.values())
    for key, value in plot_kwargs.items():
        if value is False:
            assert all(key not in child for child in pc.viz.children.values())
        elif key in ["mean", "sd", "mean_text", "sd_text"]:
            if extra_methods is False:
                assert all(key not in child for child in pc.viz.children.values())
            else:
                assert all(key in child for child in pc.viz.children.values())
        elif key == "rug":
            if rug is False:
                assert all(key not in child for child in pc.viz.children.values())
            else:
                assert all(key in child for child in pc.viz.children.values())
        else:
            assert all(key in child for child in pc.viz.children.values())


@given(
    plot_kwargs=st.fixed_dictionaries(
        {},
        optional={
            "ess_bulk": plot_kwargs_value,
            "ess_bulk_line": plot_kwargs_value,
            "ess_tail": plot_kwargs_value,
            "ess_tail_line": plot_kwargs_value,
            "xlabel": plot_kwargs_value_no_false,
            "ylabel": plot_kwargs_value_no_false,
            "mean": plot_kwargs_value,
            "mean_text": plot_kwargs_value,
            "sd": plot_kwargs_value,
            "sd_text": plot_kwargs_value,
            "min_ess": plot_kwargs_value,
            "title": plot_kwargs_value,
            "remove_axis": st.just(False),
        },
    ),
    relative=st.booleans(),
    min_ess=st.integers(min_value=10, max_value=150),
    extra_methods=st.booleans(),
    n_points=st.integers(min_value=2, max_value=12),
)
def test_plot_ess_evolution(datatree, relative, n_points, extra_methods, min_ess, plot_kwargs):
    pc = plot_ess_evolution(
        datatree,
        backend="none",
        relative=relative,
        n_points=n_points,
        extra_methods=extra_methods,
        min_ess=min_ess,
        plot_kwargs=plot_kwargs,
    )
    assert all("plot" in child for child in pc.viz.children.values())
    for key, value in plot_kwargs.items():
        if value is False:
            assert all(key not in child for child in pc.viz.children.values())
        elif key in ["mean", "sd", "mean_text", "sd_text"]:
            if extra_methods is False:
                assert all(key not in child for child in pc.viz.children.values())
            else:
                assert all(key in child for child in pc.viz.children.values())
        elif key != "remove_axis":
            assert all(key in child for child in pc.viz.children.values())
