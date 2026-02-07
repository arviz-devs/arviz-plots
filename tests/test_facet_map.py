# pylint: disable=no-self-use, redefined-outer-name
"""Test PlotCollection.facet_map() method."""
import numpy as np
import pytest
from arviz_base import dict_to_dataset

import arviz_plots as azp
from arviz_plots import PlotCollection


@pytest.fixture(scope="module")
def dataset(seed=31):
    rng = np.random.default_rng(seed)
    return dict_to_dataset(
        {
            "mu": rng.normal(size=(3, 100)),
            "tau": rng.gamma(2, 0.5, size=(3, 100)),
            "theta": rng.normal(size=(3, 100, 4)),
        },
        dims={"theta": ["school"]},
    )


class TestFacetMap:
    """Test facet_map method."""

    def test_string_function_lookup(self, dataset):
        """Test using function name as string."""
        pc = PlotCollection.grid(dataset, cols=["__variable__"], backend="none")
        result = pc.facet_map("set_xlim", limits=(-10, 10))
        assert result is pc

    def test_callable_function(self, dataset):
        """Test passing function directly."""
        pc = PlotCollection.grid(dataset, cols=["__variable__"], backend="none")
        result = pc.facet_map(azp.visuals.set_xlim, limits=(-10, 10))
        assert result is pc

    def test_filter_by_var_names(self, dataset):
        """Test applying to specific variables."""
        pc = PlotCollection.grid(dataset, cols=["__variable__"], backend="none")
        pc.facet_map("set_xlim", limits=(-5, 5), var_names="mu")
        # If this doesn't error, the filtering worked

    def test_filter_by_coords(self, dataset):
        """Test applying to specific coordinates."""
        pc = PlotCollection.grid(
            dataset[["theta"]], cols=["__variable__", "school"], backend="none"
        )
        pc.facet_map("set_xlim", limits=(-3, 3), coords={"school": [0, 1]})

    def test_var_names_and_coords_combined(self, dataset):
        """Test filtering by both var_names and coords."""
        pc = PlotCollection.grid(dataset, cols=["__variable__"], backend="none")
        pc.facet_map("set_xlim", limits=(-2, 2), var_names="theta", coords={"school": [0]})

    def test_method_chaining(self, dataset):
        """Test that method returns self for chaining."""
        pc = PlotCollection.grid(dataset, cols=["__variable__"], backend="none")
        result = pc.facet_map("set_xlim", limits=(-10, 10)).facet_map("set_ylim", limits=(0, 1))
        assert result is pc

    def test_invalid_function_name(self, dataset):
        """Test error for nonexistent function."""
        pc = PlotCollection.grid(dataset, cols=["__variable__"], backend="none")
        with pytest.raises(ValueError, match="not found in visuals module"):
            pc.facet_map("this_doesnt_exist", limits=(-10, 10))

    def test_all_variables_no_filter(self, dataset):
        """Test applying to all variables when no filter specified."""
        pc = PlotCollection.grid(dataset, cols=["__variable__"], backend="none")
        pc.facet_map("set_xlim", limits=(-20, 20))

    def test_multiple_var_names(self, dataset):
        """Test passing list of variable names."""
        pc = PlotCollection.grid(dataset, cols=["__variable__"], backend="none")
        pc.facet_map("set_xlim", limits=(-8, 8), var_names=["mu", "tau"])
