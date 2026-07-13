# pylint: disable=redefined-outer-name, import-outside-toplevel
"""Test that every bundled arviz style registers and renders on every installed backend."""

from pathlib import Path

import pytest

import arviz_plots as azp
from arviz_plots import plot_dist, style

STYLES_DIR = Path(azp.__file__).parent / "styles"
# Source of truth: every style we ship a matplotlib ``.mplstyle`` for.
# The list auto-updates when a style is added/removed.
ARVIZ_STYLES = sorted(path.stem for path in STYLES_DIR.glob("*.mplstyle"))
BACKENDS = ["matplotlib", "bokeh", "plotly"]


@pytest.fixture
def restore_style(request):
    """Revert global style state to the library defaults after each test.

    ``style.use`` sets the default style/template on *every* installed backend (not just
    the one under test), so all three are reverted to their library defaults here. Each
    reset is gated on the matching ``--skip-<backend>`` flag (like ``check_skips``) so a
    backend whose tests are skipped -- e.g. because it is not installed -- is never imported.
    No setup is needed.
    """
    yield

    if not request.config.getoption("--skip-mpl"):
        import matplotlib

        matplotlib.rcdefaults()
    if not request.config.getoption("--skip-plotly"):
        import plotly.io as pio

        pio.templates.default = "plotly"
    if not request.config.getoption("--skip-bokeh"):
        from bokeh.io import curdoc
        from bokeh.themes import default as default_bokeh_theme

        curdoc().theme = default_bokeh_theme


@pytest.mark.parametrize("style_name", ARVIZ_STYLES)
def test_style_registered_all_backends(style_name):
    available = style.available()
    for backend in BACKENDS:
        if backend in available:
            assert style_name in available[backend], f"{style_name} missing for {backend}"
    if "common" in available:
        assert style_name in available["common"]


@pytest.mark.parametrize("style_name", ARVIZ_STYLES)
@pytest.mark.matplotlib
@pytest.mark.usefixtures("restore_style", "check_skips")
def test_style_uses_default_dpi(style_name):
    mpl_style = style.get(style_name, backend="matplotlib")
    assert mpl_style["figure.dpi"] == 100


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("style_name", ARVIZ_STYLES)
@pytest.mark.usefixtures("restore_style", "clean_plots", "check_skips")
def test_style_renders(datatree, style_name, backend):
    style.use(style_name)
    pc = plot_dist(datatree, backend=backend)
    assert "figure" in pc.viz.data_vars
