# pylint: disable=no-self-use, redefined-outer-name
"""Test PlotCollection."""
import numpy as np
import pytest
from arviz_base import dict_to_dataset

from arviz_plots import PlotCollection


@pytest.fixture(scope="module")
def dataset(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(3, 10))
    theta = rng.normal(size=(3, 10, 7))
    eta = rng.normal(size=(3, 10, 4, 7))

    return dict_to_dataset(
        {"mu": mu, "theta": theta, "eta": eta},
        dims={"theta": ["hierarchy"], "eta": ["group", "hierarchy"]},
    )


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh"])
class TestFacetting:
    def test_wrap(self, dataset, backend):
        pc = PlotCollection.wrap(
            dataset[["theta", "eta"]], backend=backend, cols=["hierarchy"], col_wrap=4
        )
        assert "plot" in pc.viz.data_vars
        assert pc.viz["plot"].shape == (7,)
        assert pc.viz["row"].max() == 1
        assert pc.viz["col"].max() == 3

    def test_wrap_variable(self, dataset, backend):
        pc = PlotCollection.wrap(dataset, backend=backend, cols=["__variable__", "group"])
        assert "plot" not in pc.viz.data_vars
        assert all(f"/{var_name}" in pc.viz.groups for var_name in ("mu", "theta", "eta"))
        assert all("plot" in pc.viz[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.viz["mu"]["plot"].size == 1
        assert pc.viz["theta"]["plot"].size == 1
        assert pc.viz["eta"]["plot"].size == 4

    def test_wrap_only_variable(self, dataset, backend):
        pc = PlotCollection.wrap(dataset, backend=backend, cols=["__variable__"])
        assert "plot" not in pc.viz.data_vars
        assert all(f"/{var_name}" in pc.viz.groups for var_name in ("mu", "theta", "eta"))
        assert all("plot" in pc.viz[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.viz["mu"]["plot"].size == 1
        assert pc.viz["theta"]["plot"].size == 1
        assert pc.viz["eta"]["plot"].size == 1

    def test_grid(self, dataset, backend):
        pc = PlotCollection.grid(
            dataset[["theta", "eta"]], backend=backend, cols=["chain"], rows=["hierarchy"]
        )
        assert "plot" in pc.viz.data_vars
        assert not pc.viz.children
        assert "chain" in pc.viz["plot"].dims
        assert pc.viz["plot"].sizes["chain"] == 3
        assert "hierarchy" in pc.viz["plot"].dims
        assert pc.viz["plot"].sizes["hierarchy"] == 7
        assert "group" not in pc.viz["plot"].dims

    def test_grid_scalar(self, dataset, backend):
        pc = PlotCollection.grid(dataset, backend=backend)
        assert "plot" in pc.viz.data_vars
        assert not pc.viz.children
        assert pc.viz["plot"].size == 1

    @pytest.mark.parametrize("axis", ["rows", "cols"])
    def test_grid_rows_cols(self, dataset, backend, axis):
        pc = PlotCollection.grid(dataset[["theta", "eta"]], backend=backend, **{axis: ["chain"]})
        assert "plot" in pc.viz.data_vars
        assert not pc.viz.children
        assert "chain" in pc.viz["plot"].dims
        assert pc.viz["plot"].sizes["chain"] == 3
        assert "hierarchy" not in pc.viz["plot"].dims
        assert "group" not in pc.viz["plot"].dims
        assert pc.viz["row" if axis == "cols" else "col"].max() == 0
        assert pc.viz[axis[:3]].max() == 2

    def test_grid_variable(self, dataset, backend):
        pc = PlotCollection.grid(
            dataset[["theta", "eta"]], backend=backend, cols=["hierarchy"], rows=["__variable__"]
        )
        assert "plot" not in pc.viz.data_vars
        assert all(f"/{var_name}" in pc.viz.groups for var_name in ("theta", "eta"))
        assert all("plot" in pc.viz[var_name].data_vars for var_name in ("theta", "eta"))


class TestAesthetics:
    def test_1d_aes(self, dataset):
        pc = PlotCollection.grid(dataset)
        assert pc


class TestMap:
    def test_map(self, dataset):
        pc = PlotCollection.grid(dataset)
        assert pc
