# pylint: disable=no-self-use, redefined-outer-name
"""Test utility functions for plotting."""
import pytest
import xarray as xr

from arviz_plots.plots.utils import get_group, get_visual_kwargs, process_group_variables_coords


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
