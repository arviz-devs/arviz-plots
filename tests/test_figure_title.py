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


class TestPlotCollectionTitleMethod:
    """Test PlotCollection.add_title() method (not backend-specific)."""

    def test_add_title_method_exists(self, dataset):
        """Test that add_title method exists and is callable."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend="matplotlib",
        )
        assert hasattr(pc, "add_title")
        assert callable(pc.add_title)

    def test_add_title_returns_object(self, dataset):
        """Test that add_title returns a title object."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend="matplotlib",
        )
        result = pc.add_title("Test Title")
        assert result is not None

    def test_add_title_stores_in_viz(self, dataset):
        """Test that add_title stores the title in viz DataTree."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend="matplotlib",
        )
        pc.add_title("Test Title")
        assert "figure_title" in pc.viz

    def test_add_title_without_figure_raises(self, dataset):
        """Test that add_title raises ValueError when no figure exists."""
        pc = PlotCollection(dataset, backend="matplotlib")
        with pytest.raises(ValueError, match="No figure found"):
            pc.add_title("Test Title")


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly"])
@pytest.mark.usefixtures("clean_plots")
@pytest.mark.usefixtures("check_skips")
class TestBackendSetFigureTitle:
    """Test backend-specific set_figure_title functions."""

    def test_set_figure_title_basic(self, dataset, backend):
        """Test that set_figure_title works for each backend."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
        )
        pc.add_title("Backend Test Title")
        fig = pc.viz["figure"].item()
        self._verify_title_exists(fig, backend)

    def test_set_figure_title_with_color(self, dataset, backend):
        """Test set_figure_title with color parameter."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
        )
        result = pc.add_title("Colored Title", color="red")
        assert result is not None

    def test_set_figure_title_with_size(self, dataset, backend):
        """Test set_figure_title with size parameter."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
        )
        result = pc.add_title("Sized Title", size=16)
        assert result is not None

    def test_backward_compatibility_no_title(self, dataset, backend):
        """Test that plots work without title (backward compatibility)."""
        pc = PlotCollection.grid(
            dataset,
            cols=["__variable__"],
            backend=backend,
        )
        assert "figure" in pc.viz.data_vars
        # Should not have title if not added
        assert "figure_title" not in pc.viz

    def _verify_title_exists(self, fig, backend):
        """Verify that a title was added (basic check)."""
        if backend == "matplotlib":
            assert hasattr(fig, "_suptitle")
            assert fig._suptitle is not None
        elif backend == "plotly":
            assert fig.layout.title is not None
            assert fig.layout.title.text is not None
        elif backend == "bokeh":
            # After add_title, bokeh wraps the figure in a Column layout
            from bokeh.layouts import Column
            assert isinstance(fig, Column)
