# pylint: disable=no-self-use, redefined-outer-name
"""Test batteries-included plots using the none backend."""
import hypothesis.strategies as st
import numpy as np
import pytest
from arviz_base import from_dict
from hypothesis import given, settings

from arviz_plots import plot_dist, plot_forest

pytestmark = pytest.mark.usefixtures("no_artist_kwargs")


@pytest.fixture(scope="module")
def datatree(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(3, 50))
    tau = rng.normal(size=(3, 50, 2))
    theta = rng.normal(size=(3, 50, 2, 3))
    diverging = rng.choice([True, False], size=(3, 50), p=[0.1, 0.9])

    return from_dict(
        {
            "posterior": {"mu": mu, "theta": theta, "tau": tau},
            "sample_stats": {"diverging": diverging},
        },
        dims={"theta": ["hierarchy", "group"], "tau": ["hierarchy"]},
    )


kind_value = st.sampled_from(("kde", "ecdf"))
ci_kind_value = st.sampled_from(("eti", "hdi"))
point_estimate_value = st.sampled_from(("mean", "median"))
plot_kwargs_value = st.sampled_from(({}, False, {"color": "red"}))


@st.composite
def list_and_element(draw, elements):
    labels = draw(st.lists(elements, unique=True))
    i = draw(st.integers(min_value=-1, max_value=len(labels) - 1))
    if i == -1:
        return (labels, None)
    return (labels, labels[i])


class TestPlotsNoneBackend:
    @settings(deadline=2000)
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
    def test_plot_dist(self, datatree, kind, ci_kind, point_estimate, plot_kwargs):
        kind_kwargs = plot_kwargs.pop("kind", None)
        if kind_kwargs is not None:
            plot_kwargs[kind] = kind_kwargs
        print(plot_kwargs)
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

    @settings(deadline=2000)
    @given(
        plot_kwargs=st.fixed_dictionaries(
            {},
            optional={
                "trunk": plot_kwargs_value,
                "twig": plot_kwargs_value,
                "point_estimate": plot_kwargs_value,
                "labels": plot_kwargs_value,
                "shade": plot_kwargs_value,
                "ticklabels": st.sampled_from(({}, False)),
                "remove_axis": st.just(False),
            },
        ),
        combined=st.booleans(),
        ci_kind=ci_kind_value,
        point_estimate=point_estimate_value,
        labels_shade_label=list_and_element(
            st.sampled_from(("__variable__", "hierarchy", "group"))
        ),
    )
    def test_plot_forest(
        self, datatree, combined, ci_kind, point_estimate, plot_kwargs, labels_shade_label
    ):
        print(plot_kwargs)
        pc = plot_forest(
            datatree,
            backend="none",
            combined=combined,
            ci_kind=ci_kind,
            point_estimate=point_estimate,
            labels=labels_shade_label[0],
            shade_label=labels_shade_label[1],
            plot_kwargs=plot_kwargs,
        )
        assert all("plot" not in child for child in pc.viz.children.values())
        assert "plot" in pc.viz.data_vars
        for key, value in plot_kwargs.items():
            if value is False:
                assert all(key not in child for child in pc.viz.children.values())
            elif key != "remove_axis":
                assert all(key in child for child in pc.viz.children.values())
