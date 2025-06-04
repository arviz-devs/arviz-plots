# pylint: disable=redefined-outer-name
"""Configuration for test suite."""
import logging
import os

import pytest
from hypothesis import settings

_log = logging.getLogger("arviz_plots")

settings.register_profile("fast", deadline=2000, max_examples=20)
settings.register_profile("chron", deadline=2000, max_examples=500)
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
