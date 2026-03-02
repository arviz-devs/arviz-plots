# pylint: disable=no-self-use, redefined-outer-name
"""Tests for combine_plots — issue #236.

``combine_plots`` internally renders child plots with ``backend='none'`` before
mapping them onto a real grid, so the outer backend must be a real plotting
backend (matplotlib, bokeh, plotly) and the ``no_artist_kwargs`` fixture must be
omitted.
"""
import pytest

from arviz_plots import (
    PlotCollection,
    combine_plots,
    plot_dist,
    plot_forest,
    plot_ppc_dist,
    plot_ppc_pit,
)

pytestmark = [
    pytest.mark.usefixtures("clean_plots"),
    pytest.mark.usefixtures("check_skips"),
]


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly"])
class TestCombinePlots:
    """Test suite for combine_plots (issue #236)."""

    def test_combine_basic_column(self, datatree, backend):
        """combine_plots returns a PlotCollection with column expand (default)."""
        pc = combine_plots(
            datatree,
            plots=[
                (plot_dist, {}),
                (plot_dist, {"kind": "ecdf"}),
            ],
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars
        assert "column" in pc.viz["plot"].dims

    def test_combine_basic_row(self, datatree, backend):
        """combine_plots works with expand='row'."""
        pc = combine_plots(
            datatree,
            plots=[
                (plot_dist, {}),
                (plot_dist, {"kind": "ecdf"}),
            ],
            expand="row",
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars
        assert "row" in pc.viz["plot"].dims

    def test_combine_custom_plot_names(self, datatree, backend):
        """Custom plot_names appear as coordinate values."""
        names = ["kde_plot", "ecdf_plot"]
        pc = combine_plots(
            datatree,
            plots=[
                (plot_dist, {}),
                (plot_dist, {"kind": "ecdf"}),
            ],
            plot_names=names,
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        column_coords = list(pc.viz["plot"].coords["column"].values)
        assert column_coords == names

    def test_combine_sample_dims_forwarded(self, datatree_sample, backend):
        """sample_dims is forwarded to child plot functions without error."""
        pc = combine_plots(
            datatree_sample,
            plots=[
                (plot_dist, {}),
                (plot_dist, {"kind": "ecdf"}),
            ],
            sample_dims="sample",
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars

    def test_combine_invalid_expand(self, datatree, backend):
        """combine_plots raises ValueError for invalid expand value."""
        with pytest.raises(ValueError, match="must be 'row' or 'column'"):
            combine_plots(
                datatree,
                plots=[(plot_dist, {})],
                expand="invalid",
                backend=backend,
            )

    def test_combine_multiple_plots(self, datatree, backend):
        """Three plots can be combined in a single figure."""
        pc = combine_plots(
            datatree,
            plots=[
                (plot_dist, {}),
                (plot_dist, {"kind": "ecdf"}),
                (plot_dist, {"kind": "hist"}),
            ],
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars
        assert pc.viz["plot"].sizes["column"] == 3

    def test_combine_with_kwargs(self, datatree, backend):
        """Extra kwargs are forwarded to individual plot functions."""
        pc = combine_plots(
            datatree,
            plots=[
                (plot_dist, {"kind": "ecdf"}),
                (plot_dist, {"kind": "hist"}),
            ],
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars

    def test_combine_ppc_plots(self, datatree, backend):
        """Combine PPC-specific plots using group='posterior_predictive'."""
        pc = combine_plots(
            datatree,
            plots=[
                (plot_ppc_pit, {}),
                (plot_ppc_dist, {}),
            ],
            group="posterior_predictive",
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars

    @pytest.mark.xfail(
        reason=(
            "plot_forest uses PlotCollection.grid internally with a 'column' "
            "dimension for labels/shading, which conflicts with combine_plots's "
            "own column/row expand dimension. See issue #236."
        ),
        strict=False,
    )
    def test_combine_with_forest_xfail(self, datatree, backend):
        """Combining with plot_forest is known to fail (xfail documented bug)."""
        pc = combine_plots(
            datatree,
            plots=[
                (plot_dist, {}),
                (plot_forest, {}),
            ],
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
