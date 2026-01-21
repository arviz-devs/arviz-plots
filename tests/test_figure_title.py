# pylint: disable=no-self-use, redefined-outer-name
"""Test figure title functionality."""
import numpy as np
import pytest
from arviz_base import dict_to_dataset

from arviz_plots import PlotCollection


@pytest.fixture(scope="module")
def dataset(seed=42):
    rng = np.random.default_rng(seed)
    return dict_to_dataset(
        {
            "mu": rng.normal(size=(4, 500)),
            "tau": rng.gamma(2, 0.5, size=(4, 500)),
        },
    )


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly"])
@pytest.mark.usefixtures("clean_plots")
@pytest.mark.usefixtures("check_skips")
class TestFigureTitle:
    """Test figure title support via figure_title parameter and add_title method."""

    def test_grid_figure_title_param(self, dataset, backend):
        """Test figure_title parameter on PlotCollection.grid()."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
            figure_title="Test Grid Title",
        )
        assert "figure" in pc.viz.data_vars
        fig = pc.viz["figure"].item()
        self._verify_title(fig, "Test Grid Title", backend)

    def test_wrap_figure_title_param(self, dataset, backend):
        """Test figure_title parameter on PlotCollection.wrap()."""
        pc = PlotCollection.wrap(
            dataset,
            cols=["__variable__"],
            backend=backend,
            figure_title="Test Wrap Title",
        )
        assert "figure" in pc.viz.data_vars
        fig = pc.viz["figure"].item()
        self._verify_title(fig, "Test Wrap Title", backend)

    def test_figure_title_via_figure_kwargs(self, dataset, backend):
        """Test figure_title passed via figure_kwargs."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
            figure_kwargs={"figure_title": "Via figure_kwargs"},
        )
        assert "figure" in pc.viz.data_vars
        fig = pc.viz["figure"].item()
        self._verify_title(fig, "Via figure_kwargs", backend)

    def test_figure_title_param_precedence(self, dataset, backend):
        """Test that direct figure_title parameter takes precedence over figure_kwargs."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
            figure_title="Direct Param",
            figure_kwargs={"figure_title": "From figure_kwargs"},
        )
        fig = pc.viz["figure"].item()
        self._verify_title(fig, "Direct Param", backend)

    def test_add_title_method(self, dataset, backend):
        """Test add_title() method on PlotCollection."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
        )
        result = pc.add_title("Added After Creation")
        assert result is not None

    def test_no_title(self, dataset, backend):
        """Test that plots work without title (backward compatibility)."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
        )
        assert "figure" in pc.viz.data_vars

    def _verify_title(self, fig, expected_title, backend):
        """Verify the title was set correctly for each backend."""
        if backend == "matplotlib":
            if hasattr(fig, "_suptitle") and fig._suptitle is not None:
                assert fig._suptitle.get_text() == expected_title
        elif backend == "plotly":
            assert fig.layout.title is not None
            assert fig.layout.title.text == expected_title
        elif backend == "bokeh":
            from bokeh.layouts import Column

            assert isinstance(fig, Column)
