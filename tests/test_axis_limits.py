# pylint: disable=no-self-use, redefined-outer-name
"""Tests for PlotCollection.set_xlim() and set_ylim() methods."""
import numpy as np
import pytest
import xarray as xr

from arviz_plots import PlotCollection


@pytest.fixture(scope="module")
def sample_dataset():
    """Create a sample dataset for testing."""
    return xr.Dataset(
        {
            "mu": (["chain", "draw"], np.random.randn(4, 100)),
            "tau": (["chain", "draw"], np.abs(np.random.randn(4, 100))),
        }
    )


@pytest.fixture(scope="module")
def single_var_dataset():
    """Create a dataset with a single variable."""
    return xr.Dataset(
        {
            "mu": (["chain", "draw"], np.random.randn(4, 100)),
        }
    )


class TestSetXlim:
    """Test the set_xlim method on PlotCollection."""

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_xlim_all_plots(self, sample_dataset, backend):
        """Test setting x-axis limits for all plots."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_xlim((-10, 10))
        # Verify no errors are raised - actual limit values are backend specific

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_xlim_single_variable(self, single_var_dataset, backend):
        """Test setting x-axis limits for a single variable dataset."""
        pc = PlotCollection.grid(single_var_dataset, backend=backend)
        pc.set_xlim((0, 5))

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_xlim_with_coords(self, sample_dataset, backend):
        """Test setting x-axis limits for specific coordinates."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_xlim((-5, 5), coords={"__variable__": "mu"})

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_xlim_tuple_limits(self, sample_dataset, backend):
        """Test that limits work with tuple input."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_xlim((-1.5, 2.5))

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_xlim_list_limits(self, sample_dataset, backend):
        """Test that limits work with list input."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_xlim([-3, 3])


class TestSetYlim:
    """Test the set_ylim method on PlotCollection."""

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_ylim_all_plots(self, sample_dataset, backend):
        """Test setting y-axis limits for all plots."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_ylim((0, 1))

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_ylim_single_variable(self, single_var_dataset, backend):
        """Test setting y-axis limits for a single variable dataset."""
        pc = PlotCollection.grid(single_var_dataset, backend=backend)
        pc.set_ylim((0, 2))

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_ylim_with_coords(self, sample_dataset, backend):
        """Test setting y-axis limits for specific coordinates."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_ylim((0, 0.5), coords={"__variable__": "tau"})

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_ylim_tuple_limits(self, sample_dataset, backend):
        """Test that limits work with tuple input."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_ylim((0.1, 0.9))


class TestAxisLimitsEdgeCases:
    """Test edge cases and error handling for axis limits."""

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_limits_after_wrap(self, sample_dataset, backend):
        """Test setting limits works with wrap() created PlotCollection."""
        pc = PlotCollection.wrap(sample_dataset, backend=backend, cols=["__variable__"])
        pc.set_xlim((-5, 5))
        pc.set_ylim((0, 1))

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_limits_negative_values(self, sample_dataset, backend):
        """Test setting negative limits."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_xlim((-100, -50))
        pc.set_ylim((-10, -1))

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_limits_float_values(self, sample_dataset, backend):
        """Test setting float limits."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_xlim((-0.5, 20.5))
        pc.set_ylim((0.001, 0.999))

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_xlim_then_ylim(self, sample_dataset, backend):
        """Test setting both x and y limits sequentially."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        pc.set_xlim((-5, 5))
        pc.set_ylim((0, 1))

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_limits_nonexistent_variable(self, sample_dataset, backend):
        """Test setting limits with a variable name that doesn't exist."""
        pc = PlotCollection.grid(sample_dataset, backend=backend)
        # Should not raise error, just not apply to any plots
        pc.set_xlim((-5, 5), coords={"__variable__": "nonexistent"})

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_limits_with_chain_coord(self, sample_dataset, backend):
        """Test setting limits filtering by chain dimension."""
        pc = PlotCollection.grid(
            sample_dataset.expand_dims(chain_facet=[0, 1]), backend=backend, cols=["chain_facet"]
        )
        pc.set_xlim((-2, 2), coords={"chain_facet": 0})


class TestAxisLimitsMatplotlibSpecific:
    """Test matplotlib-specific behavior for axis limits."""

    def test_set_xlim_returns_correct_limits(self, sample_dataset):
        """Test that matplotlib axes actually have the correct limits set."""
        pc = PlotCollection.grid(sample_dataset, backend="matplotlib")
        pc.set_xlim((-7, 7))

        # Check that limits were actually set on matplotlib axes
        plot_viz = pc.viz["plot"]
        if isinstance(plot_viz, xr.DataArray):
            ax = plot_viz.values.flat[0]
        else:
            ax = list(plot_viz.data_vars.values())[0].values.flat[0]

        xlim = ax.get_xlim()
        assert xlim == (-7, 7), f"Expected (-7, 7), got {xlim}"

    def test_set_ylim_returns_correct_limits(self, sample_dataset):
        """Test that matplotlib axes actually have the correct limits set."""
        pc = PlotCollection.grid(sample_dataset, backend="matplotlib")
        pc.set_ylim((0.2, 0.8))

        plot_viz = pc.viz["plot"]
        if isinstance(plot_viz, xr.DataArray):
            ax = plot_viz.values.flat[0]
        else:
            ax = list(plot_viz.data_vars.values())[0].values.flat[0]

        ylim = ax.get_ylim()
        assert ylim == (0.2, 0.8), f"Expected (0.2, 0.8), got {ylim}"

    def test_set_xlim_with_variable_filter(self, sample_dataset):
        """Test that variable filtering works correctly with matplotlib."""
        pc = PlotCollection.grid(sample_dataset, backend="matplotlib")
        pc.set_xlim((-3, 3), coords={"__variable__": "mu"})

        # Check mu has the limits
        plot_viz = pc.viz["plot"]
        # plot_viz is a Dataset when multiple variables
        if hasattr(plot_viz, "data_vars"):
            mu_ax = plot_viz["mu"].values.flat[0]
            assert mu_ax.get_xlim() == (-3, 3)

            # tau should have default limits (not -3, 3)
            tau_ax = plot_viz["tau"].values.flat[0]
            tau_xlim = tau_ax.get_xlim()
            assert tau_xlim != (-3, 3), "tau should have default limits, not the ones set for mu"
        else:
            # Single variable case - just check it doesn't error
            pass
