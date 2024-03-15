# pylint: disable=no-self-use, redefined-outer-name
"""Test PlotCollection."""
import numpy as np
import pytest
import xarray.testing as xrt
from arviz_base import dict_to_dataset
from datatree import DataTree
from xarray import Dataset

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


def test_wrap_missing_dim(dataset):
    with pytest.raises(ValueError, match="missing"):
        PlotCollection.wrap(dataset, cols=["hierarchy"])


def test_grid_repeated_dim(dataset):
    with pytest.raises(ValueError, match="same dimension"):
        PlotCollection.grid(dataset[["theta", "eta"]], cols=["hierarchy"], rows=["hierarchy"])


class TestAesthetics:
    def test_no_aes(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt()
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert not any(pc.aes[var_name] for var_name in ("mu", "theta", "eta"))

    def test_single_1d_aes(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(aes={"color": ["hierarchy"]}, color=list(range(7)))
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["theta"]["color"].size == 7
        assert pc.aes["eta"]["color"].size == 7

    def test_single_1d_aes_cycle(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(aes={"color": ["hierarchy"]}, color=list(range(3)))
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["theta"]["color"].size == 7
        assert pc.aes["theta"]["color"].max() == 2
        assert pc.aes["eta"]["color"].size == 7
        assert pc.aes["theta"]["color"].max() == 2

    def test_multiple_1d_aes(self, dataset):
        aes_list = ("color", "linestyle", "y")
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(
            aes={"color": ["hierarchy"], "linestyle": ["group"], "y": ["hierarchy"]},
            color=[f"C{i}" for i in range(7)],
            y=np.arange(7),
            linestyle=["-", "-.", ":", "--"],
        )
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all(
            all(aes in pc.aes[var_name].data_vars for aes in aes_list)
            for var_name in ("mu", "theta", "eta")
        )
        assert all(pc.aes["mu"][aes].size == 1 for aes in aes_list)
        assert pc.aes["theta"]["color"].size == 7
        assert pc.aes["theta"]["y"].size == 7
        assert pc.aes["theta"]["linestyle"].size == 1
        assert pc.aes["eta"]["color"].size == 7
        assert pc.aes["eta"]["y"].size == 7
        assert pc.aes["eta"]["linestyle"].size == 4

    def test_single_2d_aes(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(
            aes={"color": ["group", "hierarchy"]},
            color=[f"C{i}" for i in range(7 * 4)],
        )
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["theta"]["color"].size == 7
        assert pc.aes["eta"]["color"].dims == ("group", "hierarchy")
        assert pc.aes["eta"]["color"].sizes["group"] == 4
        assert pc.aes["eta"]["color"].sizes["hierarchy"] == 7

    def test_single_3d_aes(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(
            aes={"color": ["group", "chain", "hierarchy"]},
            color=[f"C{i}" for i in range(7 * 4 * 3)],
        )
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 3
        assert pc.aes["theta"]["color"].size == 7 * 3
        assert pc.aes["eta"]["color"].dims == ("group", "chain", "hierarchy")
        assert pc.aes["eta"]["color"].sizes["group"] == 4
        assert pc.aes["eta"]["color"].sizes["chain"] == 3
        assert pc.aes["eta"]["color"].sizes["hierarchy"] == 7


def map_auxiliar(da, target, da_list, target_list, kwarg_list, **kwargs):
    da_list.append(da)
    target_list.append(target)
    kwarg_list.append(kwargs)
    return da.mean()


def generate_plot_collection1(data):
    viz_dt = DataTree.from_dict(
        {
            "/": Dataset({"chart": "chart"}),
            "mu": Dataset({"plot": "mu_plot"}),
            "theta": Dataset({"plot": "theta_plot"}),
            "eta": Dataset({"plot": "eta_plot"}),
        }
    )
    theta_color = [f"theta_C{i}" for i in range(7)]
    eta_color = [f"eta_C{i}" for i in range(7)]
    aes_dt = DataTree.from_dict(
        {
            "mu": Dataset({"color": "mu_C0"}),
            "theta": Dataset({"color": (("hierarchy",), theta_color)}),
            "eta": Dataset({"color": (("hierarchy",), eta_color)}),
        }
    )
    return PlotCollection(data, viz_dt=viz_dt, aes_dt=aes_dt, backend="backend")


class TestMap:
    def test_map(self, dataset):
        pc = generate_plot_collection1(dataset)
        assert "color" in pc.aes_set
        theta_color = list(pc.aes.theta.color.values)
        eta_color = list(pc.aes.eta.color.values)
        da_list = []
        target_list = []
        kwarg_list = []
        pc.map(
            map_auxiliar,
            "mean",
            da_list=da_list,
            target_list=target_list,
            kwarg_list=kwarg_list,
        )
        assert all(len(aux_list) == 15 for aux_list in (da_list, target_list, kwarg_list))
        assert target_list == ["mu_plot"] + ["theta_plot"] * 7 + ["eta_plot"] * 7
        assert all("color" in kwargs for kwargs in kwarg_list)
        assert [kwargs["color"] for kwargs in kwarg_list] == ["mu_C0"] + theta_color + eta_color
        assert all("chain" in da.dims for da in da_list)
        assert "group" in da_list[-1].dims
        assert all("mean" in pc.viz[var_name] for var_name in dataset.data_vars)
        assert pc.viz["mu"]["mean"].dims == ()
        assert pc.viz["theta"]["mean"].dims == ("hierarchy",)
        assert pc.viz["eta"]["mean"].dims == ("hierarchy",)

    def test_map_ignore_aes(self, dataset):
        pc = generate_plot_collection1(dataset)
        assert "color" in pc.aes_set
        da_list = []
        target_list = []
        kwarg_list = []
        pc.map(
            map_auxiliar,
            "mean",
            ignore_aes={"color"},
            da_list=da_list,
            target_list=target_list,
            kwarg_list=kwarg_list,
        )
        assert all(len(aux_list) == 3 for aux_list in (da_list, target_list, kwarg_list))
        assert target_list == ["mu_plot", "theta_plot", "eta_plot"]
        assert all("color" not in kwargs for kwargs in kwarg_list)
        assert da_list[0].dims == ("chain", "draw")
        assert da_list[1].dims == ("chain", "draw", "hierarchy")
        assert da_list[2].dims == ("chain", "draw", "group", "hierarchy")
        assert all("mean" in pc.viz[var_name] for var_name in dataset.data_vars)
        assert all(pc.viz[var_name]["mean"].size == 1 for var_name in dataset.data_vars)

    def test_map_subset_nostore(self, dataset):
        pc = generate_plot_collection1(dataset)
        da_list = []
        target_list = []
        kwarg_list = []
        pc.map(
            map_auxiliar,
            "mean",
            da_list=da_list,
            target_list=target_list,
            kwarg_list=kwarg_list,
            subset_info=True,
            store_artist=False,
        )
        assert all(len(aux_list) == 15 for aux_list in (da_list, target_list, kwarg_list))
        assert all(
            all(key in kwargs for key in ("var_name", "sel", "isel")) for kwargs in kwarg_list
        )
        assert all("mean" not in pc.viz[var_name] for var_name in dataset.data_vars)

    def test_map_over_plots(self, dataset):
        pc = generate_plot_collection1(dataset)
        theta_color = list(pc.aes.theta.color.values)
        eta_color = list(pc.aes.eta.color.values)
        da_list = []
        target_list = []
        kwarg_list = []
        pc.map(
            map_auxiliar,
            "one_per_plot",
            loop_data="plots",
            da_list=da_list,
            target_list=target_list,
            kwarg_list=kwarg_list,
        )
        assert all(len(aux_list) == 3 for aux_list in (da_list, target_list, kwarg_list))
        assert target_list == ["mu_plot", "theta_plot", "eta_plot"]
        assert all("color" in kwargs for kwargs in kwarg_list)
        assert kwarg_list[0]["color"] == "mu_C0"
        assert list(kwarg_list[1]["color"]) == theta_color
        assert list(kwarg_list[2]["color"]) == eta_color
        xrt.assert_equal(dataset["mu"], da_list[0])
        xrt.assert_equal(dataset["theta"], da_list[1])
        xrt.assert_equal(dataset["eta"], da_list[2])
        assert all("one_per_plot" in pc.viz[var_name] for var_name in dataset.data_vars)
