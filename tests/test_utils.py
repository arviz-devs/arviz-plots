# pylint: disable=no-self-use, redefined-outer-name
"""Test utility functions for plotting."""
from importlib import import_module

import numpy as np
import pytest
import xarray as xr

from arviz_plots import PlotCollection
from arviz_plots.plots.utils import (
    annotate_bin_text,
    filter_aes,
    format_coords_as_labels,
    get_group,
    get_visual_kwargs,
    process_group_variables_coords,
    set_grid_layout,
    set_wrap_layout,
)


class TestUtils:
    """Test utility functions for plotting."""

    # --- Tests for get_group ---

    def test_get_group_happy_path(self, datatree):
        """Test extracting groups from different valid inputs."""
        group = get_group(datatree, "posterior")
        assert isinstance(group, xr.Dataset)
        assert "mu" in group

        ds_input = datatree["posterior"]
        assert get_group(ds_input, "posterior") == ds_input.ds

        fake_dict = {"my_group": datatree["posterior"]}
        group_from_dict = get_group(fake_dict, "my_group")
        assert isinstance(group_from_dict, xr.Dataset)

    def test_get_group_missing(self, datatree):
        """Test behavior when group is missing (Error vs None)."""
        assert get_group(datatree, "ghost_group", allow_missing=True) is None

        with pytest.raises(KeyError):
            get_group(datatree, "ghost_group", allow_missing=False)

    # --- Tests for get_visual_kwargs ---

    def test_get_visual_kwargs(self):
        """Test visual kwargs extraction logic for all cases."""
        visuals = {"title": {"fontsize": 12, "color": "red"}}
        assert get_visual_kwargs(visuals, "title") == {"fontsize": 12, "color": "red"}

        visuals = {"legend": True}
        assert get_visual_kwargs(visuals, "legend") == {}

        visuals = {"grid": False}
        assert get_visual_kwargs(visuals, "grid") is False

        visuals = {}
        assert get_visual_kwargs(visuals, "title") == {}

        visuals = {}
        assert get_visual_kwargs(visuals, "title", default=False) is False

        visuals = {}
        assert get_visual_kwargs(visuals, "title", default=True) == {}

    # --- Tests for process_group_variables_coords ---

    def test_process_single_model_basic(self, datatree):
        """Test basic retrieval without filtering."""
        res = process_group_variables_coords(
            datatree, group="posterior", var_names=None, filter_vars=None, coords=None
        )
        assert isinstance(res, xr.Dataset)
        assert "mu" in res
        assert "theta" in res

    def test_process_single_model_filtering(self, datatree):
        """Test filtering by var_names."""
        res = process_group_variables_coords(
            datatree, group="posterior", var_names=["mu"], filter_vars=None, coords=None
        )
        assert "mu" in res
        assert "theta" not in res

    def test_process_single_model_coords(self, datatree):
        """Test slicing by coordinates."""
        coords = {"chain": [0]}
        res = process_group_variables_coords(
            datatree, group="posterior", var_names=None, filter_vars=None, coords=coords
        )
        assert res.sizes["chain"] == 1

    def test_process_multi_model_concatenation(self, datatree):
        """Test processing a dictionary of datasets."""
        dt_dict = {"model_A": datatree, "model_B": datatree}

        res = process_group_variables_coords(
            dt_dict, group="posterior", var_names=None, filter_vars=None, coords=None
        )

        assert "model" in res.dims
        assert res.sizes["model"] == 2

    def test_process_multi_model_missing_vars_error(self, datatree):
        """Test KeyError when variable is missing in all datasets."""
        dt_dict = {"model_A": datatree}

        with pytest.raises(KeyError, match="not found in any dataset"):
            process_group_variables_coords(
                dt_dict, group="posterior", var_names=["ghost_var"], filter_vars=None, coords=None
            )

    def test_process_dict_not_allowed_error(self, datatree):
        """Test ValueError when dict is passed but allowed_dict=False."""
        dt_dict = {"model_A": datatree}

        with pytest.raises(ValueError, match="Input data as dictionary not supported"):
            process_group_variables_coords(
                dt_dict,
                group="posterior",
                var_names=None,
                filter_vars=None,
                coords=None,
                allow_dict=False,
            )

    # --- Tests for filter_aes ---

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_filter_aes_basic(self, datatree, backend):
        """Test filter_aes correctly splits aesthetics and dimensions."""
        pc = PlotCollection.grid(
            datatree["posterior"].ds, backend=backend, aes={"color": ["chain"], "marker": ["draw"]}
        )
        aes_by_visuals = {"my_visual": ["color"]}
        sample_dims = ["chain", "draw"]

        artist_dims, artist_aes, ignore_aes = filter_aes(
            pc, aes_by_visuals, "my_visual", sample_dims
        )

        assert list(artist_aes) == ["color"]
        assert "marker" in ignore_aes
        assert "color" not in ignore_aes
        assert "draw" in artist_dims
        assert "chain" not in artist_dims

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_filter_aes_missing_visual(self, datatree, backend):
        """Test filter_aes returns empty aesthetics for missing visual."""
        pc = PlotCollection.grid(
            datatree["posterior"].ds, backend=backend, aes={"color": ["chain"]}
        )
        aes_by_visuals = {"other_visual": ["alpha"]}
        sample_dims = ["chain", "draw"]

        artist_dims, artist_aes, ignore_aes = filter_aes(
            pc, aes_by_visuals, "missing_visual", sample_dims
        )

        assert artist_aes == {}
        assert "color" in ignore_aes
        assert artist_dims == sample_dims

    # --- Tests for set_wrap_layout ---

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_wrap_layout(self, datatree, backend):
        """Test set_wrap_layout sets figsize correctly for wrapping columns."""
        plot_bknd = import_module(f"arviz_plots.backend.{backend}")
        ds = datatree["posterior"].ds
        pc_kwargs = {
            "figure_kwargs": {},
            "cols": ["chain"],
            "col_wrap": 2,
        }

        result = set_wrap_layout(pc_kwargs, plot_bknd, ds)

        assert result["figure_kwargs"]["figsize"] is not None
        assert result["figure_kwargs"]["figsize_units"] == "dots"
        assert result["col_wrap"] == 2

    # --- Tests for set_grid_layout ---

    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
    def test_set_grid_layout(self, datatree, backend):
        """Test set_grid_layout sets figsize for explicit grid."""
        plot_bknd = import_module(f"arviz_plots.backend.{backend}")
        ds = datatree["posterior"].ds
        pc_kwargs = {
            "figure_kwargs": {},
            "cols": ["chain"],
            "rows": [],
        }

        result = set_grid_layout(pc_kwargs, plot_bknd, ds)

        assert result["figure_kwargs"]["figsize"] is not None
        assert result["figure_kwargs"]["figsize_units"] == "dots"

    # --- Tests for format_coords_as_labels ---

    def test_format_coords_as_labels(self):
        """Test format_coords_as_labels generates correct labels."""
        data = xr.DataArray(
            np.random.randn(2, 3),
            coords={"chain": [0, 1], "draw": [0, 1, 2]},
            dims=("chain", "draw"),
        )
        labels = format_coords_as_labels(data)
        assert labels.shape == (6,)
        assert labels[0] == "0, 0"

    def test_format_coords_as_labels_skip_dims(self):
        """Test format_coords_as_labels respects skip_dims argument."""
        data = xr.DataArray(
            np.random.randn(2, 3),
            coords={"chain": [0, 1], "draw": [0, 1, 2]},
            dims=("chain", "draw"),
        )
        labels = format_coords_as_labels(data, skip_dims={"draw"})
        assert labels.shape == (2,)
        assert labels[0] == "0"

    # --- Tests for annotate_bin_text ---

    def test_annotate_bin_text(self):
        """Test annotate_bin_text formats text correctly."""
        target = []
        da = xr.DataArray(np.array([0]))

        result = annotate_bin_text(
            da=da, target=target, x=0, y=0, count_da=10, n_da=100, bin_format="{count} ({pct:.1f}%)"
        )

        assert result["string"] == "10 (10.0%)"

    def test_annotate_bin_text_zero_total(self):
        """Test annotate_bin_text handles zero total gracefully."""
        target = []
        da = xr.DataArray(np.array([0]))

        result = annotate_bin_text(
            da=da, target=target, x=0, y=0, count_da=0, n_da=0, bin_format="{count} ({pct:.1f}%)"
        )

        assert result["string"] == "0 (0.0%)"
