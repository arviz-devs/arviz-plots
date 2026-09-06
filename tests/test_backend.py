# pylint: disable=no-self-use, redefined-outer-name
"""Test backend interfacing functions."""

import ast
from importlib import import_module
from pathlib import Path

import pytest

from arviz_plots.backend.alias_utils import create_aesthetic_handlers

pytestmark = [
    pytest.mark.usefixtures("clean_plots"),
    pytest.mark.usefixtures("check_skips"),
    pytest.mark.usefixtures("no_artist_kwargs"),
]


@pytest.fixture(scope="module")
def decorated_dealiasers():
    out = {}
    for backend in ["matplotlib", "bokeh", "plotly"]:
        plot_bknd = import_module(f"arviz_plots.backend.{backend}")
        out[backend] = create_aesthetic_handlers(
            plot_bknd.get_default_aes, plot_bknd.get_background_color
        )(lambda **kwargs: kwargs)
    return out


# no dealiasing in none backend
@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly"])
class TestAlias:
    @pytest.mark.parametrize(
        "in_dict",
        [
            {"color": "C0", "marker": "C23"},
            {"facecolor": "B0", "edgecolor": "B3"},
            {"color": "B2", "linestyle": "C3"},
        ],
    )
    def test_alias_expansion(self, decorated_dealiasers, backend, in_dict):
        out_dict = decorated_dealiasers[backend](**in_dict)
        assert all(key in out_dict for key in in_dict)
        assert all(out_dict[key] != value for key, value in in_dict.items())

    @pytest.mark.parametrize(
        "in_dict",
        [
            {"color": "c0", "marker": "M2"},
            {"color": "B4", "linestyle": "linestyle_3"},
        ],
    )
    def test_passthrough(self, decorated_dealiasers, backend, in_dict):
        out_dict = decorated_dealiasers[backend](**in_dict)
        assert all(key in out_dict for key in in_dict)
        assert all(out_dict[key] == value for key, value in in_dict.items())


@pytest.mark.parametrize(
    ("func_name", "params"),
    [("hspan", ("ymin", "ymax", "target")), ("vspan", ("xmin", "xmax", "target"))],
)
def test_span_signatures_match_across_backends(func_name, params):
    """The span interfaces must take the same argument names in every backend.

    The matplotlib ``hspan`` used to name its second argument ``y_max``, so
    ``hspan(ymin=..., ymax=...)`` raised TypeError there while working on every
    other backend - and while ``vspan(xmin=..., xmax=...)`` worked everywhere.

    The backend functions are wrapped by an aliasing decorator that does not
    preserve signatures, so the definitions are read from the source instead.
    This also keeps the check working when an optional backend is not installed.
    """
    backend_dir = Path(import_module("arviz_plots.backend").__file__).parent
    for backend in ["matplotlib", "bokeh", "plotly", "none"]:
        source = backend_dir / backend / "core.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == func_name
        ]
        assert found, f"{backend} has no {func_name}"
        positional = tuple(arg.arg for arg in found[0].args.args)[: len(params)]
        assert positional == params, f"{backend}.{func_name}{positional} != {params}"
