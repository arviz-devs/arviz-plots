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
def restore_style():
    """Snapshot global style state per backend and restore it after the test.

    Applying a style mutates global state (matplotlib rcParams, the default plotly
    template and the bokeh document theme), so it must be restored to avoid leaking
    into other tests.
    """
    saved = {}
    try:
        import matplotlib as mpl

        saved["mpl"] = mpl.rcParams.copy()
    except ImportError:
        pass
    try:
        import plotly.io as pio

        saved["plotly"] = pio.templates.default
    except ImportError:
        pass
    try:
        from bokeh.io import curdoc

        saved["bokeh"] = curdoc().theme
    except ImportError:
        pass

    yield

    if "mpl" in saved:
        import matplotlib as mpl

        mpl.rcParams.update(saved["mpl"])
    if "plotly" in saved:
        import plotly.io as pio

        pio.templates.default = saved["plotly"]
    if "bokeh" in saved:
        from bokeh.io import curdoc

        curdoc().theme = saved["bokeh"]


@pytest.mark.parametrize("style_name", ARVIZ_STYLES)
def test_style_registered_all_backends(style_name):
    available = style.available()
    for backend in BACKENDS:
        if backend in available:
            assert style_name in available[backend], f"{style_name} missing for {backend}"
    if "common" in available:
        assert style_name in available["common"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("style_name", ARVIZ_STYLES)
@pytest.mark.usefixtures("restore_style", "clean_plots", "check_skips")
def test_style_renders(datatree, style_name, backend):
    pytest.importorskip(backend)
    style.use(style_name)
    pc = plot_dist(datatree, backend=backend)
    assert "figure" in pc.viz.data_vars
