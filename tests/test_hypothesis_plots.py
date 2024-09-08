# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots using the none backend."""
import arviz_stats  # pylint: disable=unused-import
import hypothesis.strategies as st
import numpy as np
import pytest
from arviz_base import from_dict
from datatree import DataTree
from hypothesis import given

from arviz_plots import plot_dist, plot_forest, plot_ridge, plot_rootogram

pytestmark = pytest.mark.usefixtures("no_artist_kwargs")


@pytest.fixture(scope="module")
def datatree(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(3, 50))
    tau = rng.normal(size=(3, 50, 2))
    theta = rng.normal(size=(3, 50, 2, 3))
    diverging = rng.choice([True, False], size=(3, 50), p=[0.1, 0.9])
    obs = rng.normal(size=(2, 3))  # hierarchy, group dims respectively
    prior_predictive = rng.normal(size=(1, 20, 2, 3))  # assuming 1 chain
    posterior_predictive = rng.normal(size=(3, 20, 2, 3))  # all chains

    dt = from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau},
            "sample_stats": {"diverging": diverging},
            "observed_data": {"obs": obs},
            "prior_predictive": {"obs": prior_predictive},
            "posterior_predictive": {"obs": posterior_predictive},
        },
        dims={
            "theta": ["hierarchy", "group"],
            "tau": ["hierarchy"],
            "obs": ["hierarchy", "group"],
        },
    )
    dt["point_estimate"] = dt.posterior.mean(("chain", "draw"))
    # TODO: should become dt.azstats.eti() after fix in arviz-stats
    post = dt.posterior.ds
    DataTree(name="trunk", parent=dt, data=post.azstats.eti(prob=0.5))
    DataTree(name="twig", parent=dt, data=post.azstats.eti(prob=0.9))
    return dt


kind_value = st.sampled_from(("kde", "ecdf"))
ci_kind_value = st.sampled_from(("eti", "hdi"))
point_estimate_value = st.sampled_from(("mean", "median"))
plot_kwargs_value = st.sampled_from(({}, False, {"color": "red"}))
ppc_root_sample_dims = st.sampled_from((["draw"], ["chain", "draw"]))
ppc_root_facet_dims = st.sampled_from((["group"], ["hierarchy"], None))


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
            "predictive": plot_kwargs_value,
            "observed": plot_kwargs_value,  # st.sampled_from(({}, {"color": "red"})),
            "observed_line": plot_kwargs_value,  # st.sampled_from(({}, {"color": "red"})),
            "baseline": plot_kwargs_value,
            "title": st.sampled_from(({}, {"color": "red"})),  # plot_kwargs_value,
            "remove_axis": st.just(False),
        },
    ),
    # group=ppc_group,
    facet_dims=ppc_root_facet_dims,
    sample_dims=ppc_root_sample_dims,
)
def test_plot_rootogram(
    datatree,
    # group,
    facet_dims,
    sample_dims,
    plot_kwargs,
):
    pc = plot_rootogram(
        datatree,
        backend="none",
        # group=group,
        facet_dims=facet_dims,
        sample_dims=sample_dims,
        plot_kwargs=plot_kwargs,
    )
    assert all("plot" in child for child in pc.viz.children.values())

    # checking presence of dimensions

    if plot_kwargs.get("predictive", {}) is not False:
        assert all(dim not in pc.viz["obs"]["predictive"].dims for dim in sample_dims)
    if plot_kwargs.get("observed", {}) is not False:
        assert all(dim not in pc.viz["obs"]["observed"].dims for dim in sample_dims)

    # checking presence of artists

    for key, value in plot_kwargs.items():
        if value is False:
            assert all(key not in child for child in pc.viz.children.values())
        elif key != "remove_axis":
            assert all(key in child for child in pc.viz.children.values())
