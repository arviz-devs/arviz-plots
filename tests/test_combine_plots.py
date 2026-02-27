# pylint: disable=no-self-use, redefined-outer-name
"""Tests for the combine_plots function.

Covers:
- Basic two-plot combination (column and row expand modes)
- Custom plot_names
- sample_dims propagation (key regression: issue #236)
- Multi-model dict input
- PlotCollection structure validation
"""
import pytest

from arviz_plots import PlotCollection, combine_plots, plot_dist, plot_trace

pytestmark = [
    pytest.mark.usefixtures("clean_plots"),
    pytest.mark.usefixtures("check_skips"),
    pytest.mark.usefixtures("no_artist_kwargs"),
]


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
class TestCombinePlots:
    """Test suite for combine_plots batteries-included function."""

    def test_combine_plots_basic(self, datatree, backend):
        """combine_plots returns a PlotCollection with expected structure."""
        pc = combine_plots(
            datatree,
            plots=[(plot_dist, {}), (plot_trace, {})],
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars
        # Default expand="column" should create a "column" dimension
        assert "column" in pc.viz.dims

    def test_combine_plots_column_names(self, datatree, backend):
        """Custom plot_names appear as coordinate values in the column dimension."""
        names = ["dist_panel", "trace_panel"]
        pc = combine_plots(
            datatree,
            plots=[(plot_dist, {}), (plot_trace, {})],
            plot_names=names,
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        col_coords = list(pc.viz.coords["column"].values)
        assert col_coords == names

    def test_combine_plots_expand_row(self, datatree, backend):
        """expand='row' places plots along a 'row' dimension instead of 'column'."""
        pc = combine_plots(
            datatree,
            plots=[(plot_dist, {}), (plot_trace, {})],
            expand="row",
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "row" in pc.viz.dims

    def test_combine_plots_expand_invalid(self, datatree, backend):
        """expand with an invalid value raises a ValueError."""
        with pytest.raises(ValueError, match="`expand` must be 'row' or 'column'"):
            combine_plots(
                datatree,
                plots=[(plot_dist, {}), (plot_trace, {})],
                expand="diagonal",
                backend=backend,
            )

    def test_combine_plots_sample_dims_propagated(self, datatree_sample, backend):
        """sample_dims passed to combine_plots is forwarded to inner plot functions.

        Regression test for https://github.com/arviz-devs/arviz-plots/issues/236.
        When sample_dims is non-default (e.g. 'sample' instead of ['chain', 'draw']),
        the inner plotting functions must receive the same value so faceting and
        dimension reduction are consistent across all sub-plots.
        """
        pc = combine_plots(
            datatree_sample,
            plots=[(plot_dist, {}), (plot_trace, {})],
            sample_dims="sample",
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        # With sample_dims="sample", chain/draw dims should NOT appear in the output
        for viz_group in pc.viz.children.values():
            for var_data in viz_group.data_vars.values():
                assert "chain" not in var_data.dims, (
                    f"'chain' dim found in viz output — sample_dims was not propagated correctly"
                )
                assert "draw" not in var_data.dims, (
                    f"'draw' dim found in viz output — sample_dims was not propagated correctly"
                )

    def test_combine_plots_var_names_filter(self, datatree, backend):
        """var_names restricts which variables appear in the combined output."""
        pc = combine_plots(
            datatree,
            plots=[(plot_dist, {}), (plot_trace, {})],
            var_names=["mu"],
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        # Only 'mu' should be present; 'tau', 'theta' should not
        for child_name, child_ds in pc.viz.children.items():
            # Skip non-variable group children
            if child_name in {"plot", "row_index", "col_index", "figure"}:
                continue
            var_names_in_child = list(child_ds.data_vars)
            for var in var_names_in_child:
                assert var == "mu", (
                    f"Unexpected variable '{var}' in viz group '{child_name}' "
                    f"after filtering to var_names=['mu']"
                )

    def test_combine_plots_two_models(self, datatree, datatree2, backend):
        """combine_plots works with a dict of models as input.

        Note: var_names is restricted to 'mu' to stay under the subplot limit
        that would otherwise be hit when combining 2 models × 2 plots × all variables.
        The 'none' backend is skipped because it cannot render multi-model dict input.
        """
        if backend == "none":
            pytest.skip("none backend does not support multi-model dict input in combine_plots")
        pc = combine_plots(
            {"model_a": datatree, "model_b": datatree2},
            plots=[(plot_dist, {}), (plot_trace, {})],
            var_names=["mu"],
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars
