# pylint: disable=no-self-use, redefined-outer-name
"""Test PlotCollection."""
import numpy as np
import pytest
import xarray.testing as xrt
from arviz_base import dict_to_dataset, load_arviz_data
from xarray import DataArray, Dataset, DataTree, concat, full_like

from arviz_plots import PlotCollection
from arviz_plots.plot_collection import _get_aes_dict_from_dt


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


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly"])
@pytest.mark.usefixtures("clean_plots")
@pytest.mark.usefixtures("check_skips")
class Testfaceting:
    def test_wrap(self, dataset, backend):
        pc = PlotCollection.wrap(
            dataset[["theta", "eta"]], backend=backend, cols=["hierarchy"], col_wrap=4
        )
        assert "plot" in pc.viz.data_vars
        assert pc.viz["plot"].shape == (7,)
        assert pc.viz["row_index"].max() == 1
        assert pc.viz["col_index"].max() == 3

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
        assert pc.viz["row_index" if axis == "cols" else "col_index"].max() == 0
        assert pc.viz["col_index" if axis == "cols" else "row_index"].max() == 2

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

    def test_single_1d_aes_neutral(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        colors = np.arange(8)
        pc.generate_aes_dt(aes={"color": ["hierarchy"]}, color=colors)
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["mu"]["color"].item() == 0
        assert pc.aes["theta"]["color"].size == 7
        assert all(pc.aes["theta"]["color"].to_numpy() == colors[1:])
        assert pc.aes["eta"]["color"].size == 7
        assert all(pc.aes["eta"]["color"].to_numpy() == colors[1:])

    def test_single_1d_aes_no_neutral(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        colors = np.arange(3)
        var_names = ("mu", "theta", "eta")
        pc.generate_aes_dt(aes={"color": ["chain"]}, color=colors)
        assert all(f"/{var_name}" in pc.aes.groups for var_name in var_names)
        assert all("color" in pc.aes[var_name].data_vars for var_name in var_names)
        assert all(pc.aes[var_name]["color"].size == 3 for var_name in var_names)
        assert all(all(pc.aes[var_name]["color"].to_numpy() == colors) for var_name in var_names)

    def test_single_1d_aes_cycle(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(aes={"color": ["hierarchy"]}, color=list(range(3)))
        color_cycle = np.array([1, 2, 1, 2, 1, 2, 1])
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["mu"]["color"].item() == 0
        assert pc.aes["theta"]["color"].size == 7
        assert np.allclose(pc.aes["theta"]["color"], color_cycle)
        assert pc.aes["eta"]["color"].size == 7
        assert np.allclose(pc.aes["eta"]["color"], color_cycle)

    def test_single_1d_aes_neutral_element_in_cycle(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(aes={"color": ["hierarchy"]}, color=[0, 0, 1, 2])
        color_cycle = np.array([0, 1, 2, 0, 1, 2, 0])
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["mu"]["color"].item() == 0
        assert pc.aes["theta"]["color"].size == 7
        assert np.allclose(pc.aes["theta"]["color"], color_cycle)
        assert pc.aes["theta"]["color"].max() == 2
        assert pc.aes["eta"]["color"].size == 7
        assert np.allclose(pc.aes["eta"]["color"], color_cycle)

    def test_single_1d_aes_variable(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(aes={"color": ["__variable__"]}, color=list(range(3)))
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert all(pc.aes[var_name]["color"].size == 1 for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].item() == 0
        assert pc.aes["theta"]["color"].item() == 1
        assert pc.aes["eta"]["color"].item() == 2

    def test_multiple_1d_aes(self, dataset):
        aes_list = ("color", "linestyle", "y")
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(
            aes={
                "color": ["hierarchy"],
                "linestyle": ["group"],
                "y": ["__variable__", "hierarchy"],
            },
            color=[f"C{i}" for i in range(7)],
            y=np.arange(1 + 7 + 7),
            linestyle=["-", "-.", ":", "--"],
        )
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all(
            all(aes in pc.aes[var_name].data_vars for aes in aes_list)
            for var_name in ("mu", "theta", "eta")
        )
        assert all(pc.aes["mu"][aes].size == 1 for aes in aes_list)
        assert pc.aes["mu"]["y"] == 0
        assert pc.aes["theta"]["color"].size == 7
        assert pc.aes["theta"]["y"].size == 7
        assert pc.aes["theta"]["y"].min() == 1
        assert pc.aes["theta"]["y"].max() == 7
        assert pc.aes["theta"]["linestyle"].size == 1
        assert pc.aes["eta"]["color"].size == 7
        assert pc.aes["eta"]["y"].size == 7
        assert pc.aes["eta"]["y"].min() == 8
        assert pc.aes["eta"]["y"].max() == 14
        assert pc.aes["eta"]["linestyle"].size == 4

    def test_single_2d_aes(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(
            aes={"color": ["group", "hierarchy"]},
            color=np.arange(7 * 4 + 1),
        )
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["theta"]["color"].size == 1
        assert all(pc.aes[var_name]["color"] == 0 for var_name in ("mu", "theta"))
        assert pc.aes["eta"]["color"].dims == ("group", "hierarchy")
        assert pc.aes["eta"]["color"].sizes["group"] == 4
        assert pc.aes["eta"]["color"].sizes["hierarchy"] == 7
        assert pc.aes["eta"]["color"].min() == 1

    def test_single_3d_aes(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(
            aes={"color": ["group", "chain", "hierarchy"]},
            color=np.arange(7 * 4 * 3 + 1),
        )
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["theta"]["color"].size == 1
        assert all(pc.aes[var_name]["color"] == 0 for var_name in ("mu", "theta"))
        assert pc.aes["eta"]["color"].dims == ("group", "chain", "hierarchy")
        assert pc.aes["eta"]["color"].sizes["group"] == 4
        assert pc.aes["eta"]["color"].sizes["chain"] == 3
        assert pc.aes["eta"]["color"].sizes["hierarchy"] == 7
        assert pc.aes["eta"]["color"].min() == 1

    def test_multiple_aes_mix(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(
            aes={
                "color": ["group", "chain", "hierarchy"],
                "linestyle": ["chain", "hierarchy"],
                "y": False,
            },
            color=np.arange(7 * 4 * 3 + 1),
            linestyle=["-", "--", ":"],
        )
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("y" not in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert all("color" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["color"].size == 1
        assert pc.aes["theta"]["color"].size == 1
        assert all(pc.aes[var_name]["color"] == 0 for var_name in ("mu", "theta"))
        assert pc.aes["eta"]["color"].dims == ("group", "chain", "hierarchy")
        assert pc.aes["eta"]["color"].sizes["group"] == 4
        assert pc.aes["eta"]["color"].sizes["chain"] == 3
        assert pc.aes["eta"]["color"].sizes["hierarchy"] == 7
        assert pc.aes["eta"]["color"].min() == 1
        assert all("linestyle" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["linestyle"].size == 1
        assert pc.aes["mu"]["linestyle"] == "-"
        assert all(
            pc.aes[var_name]["linestyle"].dims == ("chain", "hierarchy")
            for var_name in ("theta", "eta")
        )
        assert all(
            pc.aes[var_name]["linestyle"].sizes["chain"] == 3 for var_name in ("theta", "eta")
        )
        assert all(
            pc.aes[var_name]["linestyle"].sizes["hierarchy"] == 7 for var_name in ("theta", "eta")
        )
        assert all("-" not in pc.aes[var_name]["linestyle"] for var_name in ("theta", "eta"))

    def test_aes_variable_all_dims(self, dataset):
        pc = PlotCollection(dataset, DataTree())
        pc.generate_aes_dt(
            aes={"y": ["__variable__", "chain", "group", "hierarchy"]},
            y=np.arange(3 + 3 * 7 + 3 * 7 * 4),
        )
        assert all(f"/{var_name}" in pc.aes.groups for var_name in ("mu", "theta", "eta"))
        assert all("y" in pc.aes[var_name].data_vars for var_name in ("mu", "theta", "eta"))
        assert pc.aes["mu"]["y"].dims == ("chain",)
        assert pc.aes["mu"]["y"].min() == 0
        assert pc.aes["mu"]["y"].max() == 2
        assert pc.aes["theta"]["y"].dims == ("chain", "hierarchy")
        assert pc.aes["theta"]["y"].min() == 3
        assert pc.aes["theta"]["y"].max() == (3 + 3 * 7 - 1)
        assert pc.aes["eta"]["y"].dims == ("chain", "group", "hierarchy")
        assert pc.aes["eta"]["y"].min() == (3 + 3 * 7)
        assert pc.aes["eta"]["y"].max() == (3 + 3 * 7 + 3 * 7 * 4 - 1)


class TestSaveFigures:
    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param("matplotlib", id="matplotlib"),
            # See https://github.com/plotly/Kaleido/issues/194
            pytest.param(
                "plotly",
                marks=pytest.mark.skip(reason="plotly savefig is not working with pytest"),
                id="plotly",
            ),
            pytest.param("bokeh", id="bokeh"),
            pytest.param("none", id="none"),
        ],
    )
    def test_save_figures(self, dataset, backend, tmp_path):
        pc = PlotCollection.grid(dataset, backend=backend)
        if backend == "bokeh":
            ext = "html"
        else:
            ext = "png"
        file_path = tmp_path / f"test_savefig_{backend}.{ext}"

        if backend == "none":
            with pytest.raises(TypeError):
                pc.savefig(str(file_path))
        else:
            pc.savefig(str(file_path))
            assert file_path.exists()
            assert file_path.stat().st_size > 0


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


def generate_plot_collection2():
    centered = load_arviz_data("centered_eight")
    non_centered = load_arviz_data("non_centered_eight")
    data = {"centered": centered.posterior.ds, "non_centered": non_centered.posterior.ds}
    coords = {"model": list(data), "school": centered.posterior.school}
    data = concat(data.values(), dim="model").assign_coords(coords)
    theta_plot = [f"theta_plot{i}" for i in range(8)]
    theta_t_plot = [f"theta_t_plot{i}" for i in range(8)]
    viz_dt = DataTree.from_dict(
        {
            "/": Dataset({"chart": "chart"}),
            "mu": Dataset({"plot": "mu_plot"}),
            "tau": Dataset({"plot": "tau_plot"}),
            "theta": Dataset({"plot": (("school",), theta_plot)}),
            "theta_t": Dataset({"plot": (("school",), theta_t_plot)}),
        }
    )
    theta_color = [f"theta_C{i}" for i in range(8)]
    theta_t_color = [f"theta_t_C{i}" for i in range(8)]
    model_aes_dict = {"y": (("model",), [0, 1])}
    aes_dt = DataTree.from_dict(
        {
            "mu": Dataset({"color": "mu_C0", **model_aes_dict}, coords=coords),
            "tau": Dataset({"color": "tau_C0", **model_aes_dict}, coords=coords),
            "theta": Dataset(
                {"color": (("school",), theta_color), **model_aes_dict}, coords=coords
            ),
            "theta_t": Dataset(
                {"color": (("school",), theta_t_color), **model_aes_dict}, coords=coords
            ),
        }
    )
    return PlotCollection(data, viz_dt=viz_dt, aes_dt=aes_dt, backend="backend")


def test_aes_dict_from_dt_dim_neutral():
    color = [f"C{i+1}" for i in range(8)]
    aes_dt = DataTree.from_dict(
        {
            "mu": Dataset({"color": "C0"}),
            "theta": Dataset({"color": (("hierarchy",), color)}),
            "eta": Dataset({"color": (("hierarchy",), color)}),
        }
    )
    aes_dict = _get_aes_dict_from_dt(aes_dt)
    assert aes_dict == {"color": ["hierarchy"]}


def test_aes_dict_from_dt_dim_no_neutral():
    color = [f"C{i+1}" for i in range(8)]
    aes_dt = DataTree.from_dict(
        {
            "theta": Dataset({"color": (("hierarchy",), color)}),
            "eta": Dataset({"color": (("hierarchy",), color)}),
        }
    )
    aes_dict = _get_aes_dict_from_dt(aes_dt)
    assert aes_dict == {"color": ["hierarchy"]}


def test_aes_dict_from_dt_variable_dim():
    theta_color = [f"C{i+1}" for i in range(8)]
    eta_color = [f"C{i+8}" for i in range(8)]
    aes_dt = DataTree.from_dict(
        {
            "mu": Dataset({"color": "C0"}),
            "theta": Dataset({"color": (("hierarchy",), theta_color)}),
            "eta": Dataset({"color": (("hierarchy",), eta_color)}),
        }
    )
    aes_dict = _get_aes_dict_from_dt(aes_dt)
    assert aes_dict == {"color": ["__variable__", "hierarchy"]}


def test_aes_dict_from_dt_variable_dims():
    rng = np.random.default_rng(31)
    theta_y = rng.normal(size=(4, 8))
    eta_y = rng.normal(size=8)
    aes_dt = DataTree.from_dict(
        {
            "mu": Dataset({"y": 0}),
            "theta": Dataset(
                {
                    "y": (
                        (
                            "group",
                            "hierarchy",
                        ),
                        theta_y,
                    )
                }
            ),
            "eta": Dataset({"y": (("hierarchy",), eta_y)}),
        }
    )
    aes_dict = _get_aes_dict_from_dt(aes_dt)
    assert len(aes_dict) == 1
    assert "y" in aes_dict
    assert set(aes_dict["y"]) == {"__variable__", "group", "hierarchy"}


def test_aes_dict_from_dt_variable():
    aes_dt = DataTree.from_dict(
        {
            "mu": Dataset({"color": "C0"}),
            "theta": Dataset({"color": "C1"}),
            "eta": Dataset({"color": "C2"}),
        }
    )
    aes_dict = _get_aes_dict_from_dt(aes_dt)
    assert aes_dict == {"color": ["__variable__"]}


def test_aes_dataset_manipulation(dataset):
    pc = generate_plot_collection1(dataset)
    color_ds = pc.get_aes_as_dataset("color")
    assert isinstance(color_ds, Dataset)
    assert list(color_ds.data_vars) == ["mu", "theta", "eta"]
    color_ds["mu"] = "C0"
    color_ds["theta"] = "C1"
    color_ds["eta"] = full_like(color_ds["eta"], "C2")
    pc.update_aes_from_dataset("color", color_ds)
    aes_dt = pc.aes
    assert isinstance(aes_dt, DataTree)
    assert all(f"/{var_name}" in aes_dt.groups for var_name in ("mu", "theta", "eta"))
    assert all("color" in child.data_vars for child in aes_dt.children.values())
    assert aes_dt["mu"]["color"].item() == "C0"
    assert "hierarchy" not in aes_dt["theta"]["color"].dims
    assert aes_dt["theta"]["color"].item() == "C1"
    assert "hierarchy" in aes_dt["eta"]["color"].dims
    assert np.all(aes_dt["eta"]["color"] == "C2")


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

    def test_map_nan(self):
        pc = generate_plot_collection2()
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
        assert all(len(aux_list) == 28 for aux_list in (da_list, target_list, kwarg_list))
        theta_t_line = pc.viz["theta_t"]["mean"]
        assert "model" in theta_t_line.dims
        assert "school" in theta_t_line.dims
        assert all(value is None for value in theta_t_line.sel(model="centered").values.flatten())

    def test_map_xarray_kwargs(self, dataset):
        pc = generate_plot_collection1(dataset)
        ds = dataset.copy()
        da_scalar = dataset["mu"].copy()
        da_hierarchy = dataset["eta"].copy()
        assert "color" in pc.aes_set
        da_list = []
        target_list = []
        kwarg_list = []
        pc.map(
            map_auxiliar,
            "mean",
            da_list=da_list,
            target_list=target_list,
            kwarg_list=kwarg_list,
            ds=ds,
            da_scalar=da_scalar,
            da_hierarchy=da_hierarchy,
        )
        assert all(len(aux_list) == 15 for aux_list in (da_list, target_list, kwarg_list))
        assert all(
            all(key in kwargs for key in ("da_scalar", "da_hierarchy", "ds"))
            for kwargs in kwarg_list
        )
        xrt.assert_equal(kwarg_list[0]["ds"], ds["mu"])
        xrt.assert_equal(kwarg_list[-1]["ds"], ds["eta"].isel(hierarchy=-1))
        for kwargs in kwarg_list:
            xrt.assert_equal(kwargs["da_scalar"], da_scalar)
            assert isinstance(kwargs["ds"], DataArray)
            assert "school" not in kwargs["ds"].dims
            assert "school" not in kwargs["da_hierarchy"].dims
