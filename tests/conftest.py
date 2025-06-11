# pylint: disable=redefined-outer-name
"""Configuration for test suite."""
import logging
import os

import pytest
from arviz_base.testing import cmp as _cmp
from arviz_base.testing import datatree as _datatree
from arviz_base.testing import datatree2 as _datatree2
from arviz_base.testing import datatree3 as _datatree3
from arviz_base.testing import datatree_4d as _datatree_4d
from arviz_base.testing import datatree_binary as _datatree_binary
from arviz_base.testing import datatree_sample as _datatree_sample
from hypothesis import settings

_log = logging.getLogger("arviz_plots")

settings.register_profile("fast", deadline=3000, max_examples=20)
settings.register_profile("chron", deadline=3000, max_examples=500)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "fast"))


def pytest_addoption(parser):
    """Definition for command line option to save figures from tests or skip backends."""
    parser.addoption("--save", nargs="?", const="test_images", help="Save images rendered by plot")
    parser.addoption("--skip-mpl", action="store_const", const=True, help="Skip matplotlib tests")
    parser.addoption("--skip-bokeh", action="store_const", const=True, help="Skip bokeh tests")
    parser.addoption("--skip-plotly", action="store_const", const=True, help="Skip plotly tests")


@pytest.fixture(scope="session")
def save_figs(request):
    """Enable command line switch for saving generation figures upon testing."""
    fig_dir = request.config.getoption("--save")

    if fig_dir is not None:
        # Try creating directory if it doesn't exist
        _log.info("Saving generated images in %s", fig_dir)

        os.makedirs(fig_dir, exist_ok=True)
        _log.info("Directory %s created", fig_dir)

        # Clear all files from the directory
        # Does not alter or delete directories
        for file in os.listdir(fig_dir):
            full_path = os.path.join(fig_dir, file)

            try:
                os.remove(full_path)

            except OSError:
                _log.info("Failed to remove %s", full_path)

    return fig_dir


@pytest.fixture(scope="function")
def clean_plots(request, save_figs):
    """Close plots after each test, saving too if --save is specified during test invocation."""

    def fin():
        if ("backend" in request.fixturenames) and any(
            "matplotlib" in key for key in request.keywords.keys()
        ):
            import matplotlib.pyplot as plt

            if save_figs is not None:
                plt.savefig(f"{os.path.join(save_figs, request.node.name)}.png")
            plt.close("all")

    request.addfinalizer(fin)


@pytest.fixture(scope="function")
def check_skips(request):
    """Skip bokeh or matplotlib tests if requested via command line."""
    skip_mpl = request.config.getoption("--skip-mpl")
    skip_bokeh = request.config.getoption("--skip-bokeh")
    skip_plotly = request.config.getoption("--skip-plotly")

    if "backend" in request.fixturenames:
        if skip_mpl and any("matplotlib" in key for key in request.keywords.keys()):
            pytest.skip(reason="Requested skipping matplolib tests via command line argument")
        if skip_bokeh and any("bokeh" in key for key in request.keywords.keys()):
            pytest.skip(reason="Requested skipping bokeh tests via command line argument")
        if skip_plotly and any("plotly" in key for key in request.keywords.keys()):
            pytest.skip(reason="Requested skipping plotly tests via command line argument")


@pytest.fixture(scope="function")
def no_artist_kwargs(monkeypatch):
    """Raise an error if visual kwargs are present when using 'none' backend."""
    monkeypatch.setattr("arviz_plots.backend.none.ALLOW_KWARGS", False)


@pytest.fixture(scope="session")
def datatree():
    """Fixture for a general DataTree."""
    return _datatree()


@pytest.fixture(scope="session")
def datatree2():
    """Fixture for a DataTree with a posterior and sample stats."""
    return _datatree2()


@pytest.fixture(scope="session")
def datatree3():
    """Fixture for a DataTree with discrete data."""
    return _datatree3()


@pytest.fixture(scope="session")
def datatree_4d():
    """Fixture for a DataTree with a 4D posterior."""
    return _datatree_4d()


@pytest.fixture(scope="session")
def datatree_binary():
    """Fixture for a DataTree with binary data."""
    return _datatree_binary()


@pytest.fixture(scope="session")
def datatree_sample():
    """Fixture for a DataTree with sample stats."""
    return _datatree_sample()


@pytest.fixture(scope="session")
def cmp():
    """Fixture for the cmp function."""
    return _cmp()
