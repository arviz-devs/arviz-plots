# pylint: disable=no-self-use, redefined-outer-name
"""Test backend interfacing functions."""
from importlib import import_module

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
