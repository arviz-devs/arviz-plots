# pylint: disable=no-self-use, redefined-outer-name
"""Test PlotMatrix."""
import numpy as np
import pytest
import xarray as xr
from arviz_base import dict_to_dataset

from arviz_plots import PlotMatrix
from arviz_plots.plot_matrix import subset_matrix_da


@pytest.fixture(scope="module")
def dataset(seed=31):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(2, 7))
    theta = rng.normal(size=(2, 7, 2))
    eta = rng.normal(size=(2, 7, 3, 2))

    return dict_to_dataset(
        {"mu": mu, "theta": theta, "eta": eta},
        dims={"theta": ["hierarchy"], "eta": ["group", "hierarchy"]},
    )


@pytest.fixture(scope="module")
def matrix_da(seed=31):
    rng = np.random.default_rng(seed)
    var_coord = ["mu", "theta", "theta", "theta", "tau"]
    hierarchy_coord = [None, "a", "b", "c", None]
    return xr.DataArray(
        rng.normal(size=(5, 5)),
        dims=["row_index", "col_index"],
        coords={
            "var_name_x": (("col_index",), var_coord),
            "var_name_y": (("row_index",), var_coord),
            "hierarchy_x": (("col_index",), hierarchy_coord),
            "hierarchy_y": (("row_index",), hierarchy_coord),
        },
    )


@pytest.mark.parametrize("subset", (["mu", {}], ["theta", {"hierarchy": "b"}]))
def test_subset_matrix_da_diag(matrix_da, subset):
    da_subset = subset_matrix_da(matrix_da, subset[0], subset[1])
    assert not isinstance(da_subset, xr.DataArray)


@pytest.mark.parametrize("subset_x", (["mu", {}], ["theta", {"hierarchy": "b"}]))
@pytest.mark.parametrize("subset_y", (["tau", {}], ["theta", {"hierarchy": "c"}]))
def test_subset_matrix_da_offdiag(matrix_da, subset_x, subset_y):
    da_subset = subset_matrix_da(
        matrix_da, subset_x[0], subset_x[1], var_name_y=subset_y[0], selection_y=subset_y[1]
    )
    assert not isinstance(da_subset, xr.DataArray)


def test_plot_matrix_init(dataset):
    pc = PlotMatrix(dataset, ["__variable__", "hierarchy", "group"], backend="none")
    assert "plot" in pc.viz.data_vars
    coord_names = ("var_name_x", "var_name_y", "hierarchy_x", "hierarchy_y", "group_x", "group_y")
    missing_coord_names = [name for name in coord_names if name not in pc.viz["plot"].coords]
    assert not missing_coord_names, list(pc.viz["plot"].coords)
    assert pc.viz["plot"].sizes == {"row_index": 9, "col_index": 9}


def test_plot_matrix_aes(dataset):
    pc = PlotMatrix(
        dataset, ["__variable__", "hierarchy", "group"], backend="none", aes={"color": ["chain"]}
    )
    assert "/color" in pc.aes.groups
    assert "mapping" in pc.aes["color"].data_vars
    assert "neutral_element" not in pc.aes["color"].data_vars


# pylint: disable=unused-argument
def map_auxiliar(da, target, target_list, kwarg_list, **kwargs):
    target_list.append(target)
    kwarg_list.append(kwargs)
    return 1


# pylint: disable=unused-argument
def map_auxiliar_couple(da_x, da_y, target, target_list, kwarg_list, **kwargs):
    target_list.append(target)
    kwarg_list.append(kwargs)
    return 1


def test_plot_matrix_map(dataset):
    pc = PlotMatrix(
        dataset, ["__variable__", "hierarchy", "group"], backend="none", aes={"color": ["chain"]}
    )
    target_list = []
    kwarg_list = []
    pc.map(
        map_auxiliar,
        "aux",
        target_list=target_list,
        kwarg_list=kwarg_list,
    )
    assert all(len(aux_list) == 9 * 2 for aux_list in (target_list, kwarg_list))
    assert pc.viz["aux"].dims == ("row_index", "col_index", "chain")
    for i in range(9):
        for j in range(9):
            if i == j:
                assert all(
                    elem is not None for elem in pc.viz["aux"].sel(row_index=i, col_index=j).values
                )
            else:
                assert all(
                    elem is None for elem in pc.viz["aux"].sel(row_index=i, col_index=j).values
                )


def test_plot_matrix_map_scalar_coord(dataset):
    pc = PlotMatrix(
        dataset.isel(hierarchy=[0]),
        ["__variable__", "hierarchy", "group"],
        backend="none",
        aes={"color": ["chain"]},
    )
    target_list = []
    kwarg_list = []
    pc.map(
        map_auxiliar,
        "aux",
        target_list=target_list,
        kwarg_list=kwarg_list,
    )
    assert all(len(aux_list) == 5 * 2 for aux_list in (target_list, kwarg_list))
    assert pc.viz["aux"].dims == ("row_index", "col_index", "chain")
    for i in range(5):
        for j in range(5):
            if i == j:
                assert all(
                    elem is not None for elem in pc.viz["aux"].sel(row_index=i, col_index=j).values
                )
            else:
                assert all(
                    elem is None for elem in pc.viz["aux"].sel(row_index=i, col_index=j).values
                )


@pytest.mark.parametrize("triangle", ("both", "lower", "upper"))
def test_plot_matrix_map_triangle(dataset, triangle):
    pc = PlotMatrix(
        dataset, ["__variable__", "hierarchy", "group"], backend="none", aes={"color": ["chain"]}
    )
    target_list = []
    kwarg_list = []
    pc.map_triangle(
        map_auxiliar_couple,
        "aux",
        target_list=target_list,
        kwarg_list=kwarg_list,
        triangle=triangle,
    )
    aux_len = sum(range(9)) * 2
    if triangle == "both":
        aux_len *= 2
    assert all(len(aux_list) == aux_len for aux_list in (target_list, kwarg_list))
    assert pc.viz["aux"].dims == ("row_index", "col_index", "chain")
    for i in range(9):
        for j in range(9):
            is_none = (elem is None for elem in pc.viz["aux"].sel(row_index=i, col_index=j).values)
            if i == j:
                assert all(is_none)
            elif i > j:
                if triangle in ("both", "lower"):
                    assert not any(is_none)
                else:
                    assert all(is_none)
            else:
                if triangle in ("both", "upper"):
                    assert not any(is_none)
                else:
                    assert all(is_none)


@pytest.mark.parametrize("triangle", ("both", "lower", "upper"))
def test_plot_matrix_map_triangle_scalar_coord(dataset, triangle):
    pc = PlotMatrix(
        dataset.isel(hierarchy=[0]),
        ["__variable__", "hierarchy", "group"],
        backend="none",
        aes={"color": ["chain"]},
    )
    target_list = []
    kwarg_list = []
    pc.map_triangle(
        map_auxiliar_couple,
        "aux",
        target_list=target_list,
        kwarg_list=kwarg_list,
        triangle=triangle,
    )
    aux_len = sum(range(5)) * 2
    if triangle == "both":
        aux_len *= 2
    assert all(len(aux_list) == aux_len for aux_list in (target_list, kwarg_list))
    assert pc.viz["aux"].dims == ("row_index", "col_index", "chain")
    for i in range(5):
        for j in range(5):
            is_none = (elem is None for elem in pc.viz["aux"].sel(row_index=i, col_index=j).values)
            if i == j:
                assert all(is_none)
            elif i > j:
                if triangle in ("both", "lower"):
                    assert not any(is_none)
                else:
                    assert all(is_none)
            else:
                if triangle in ("both", "upper"):
                    assert not any(is_none)
                else:
                    assert all(is_none)
