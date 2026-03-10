# pylint: disable=no-self-use, redefined-outer-name
"""Tests for plot_ts — time series plot (issue #322)."""
import numpy as np
import pytest
import xarray as xr

from arviz_plots import PlotCollection, plot_ts

pytestmark = [
    pytest.mark.usefixtures("clean_plots"),
    pytest.mark.usefixtures("check_skips"),
    pytest.mark.usefixtures("no_artist_kwargs"),
]

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

N_TIME = 20
N_HOLDOUT = 5
N_CHAIN = 2
N_DRAW = 40
RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def ts_datatree():
    """Minimal DataTree with observed_data and posterior_predictive groups."""
    obs_y = RNG.normal(size=N_TIME)
    pp_y = RNG.normal(size=(N_CHAIN, N_DRAW, N_TIME))

    obs = xr.Dataset({"y": ("time", obs_y), "y_obs": ("time", obs_y)})
    pp = xr.Dataset(
        {
            "y_hat": (["chain", "draw", "time"], pp_y),
            "y_forecasts": (
                ["chain", "draw", "time_holdout"],
                RNG.normal(size=(N_CHAIN, N_DRAW, N_HOLDOUT)),
            ),
        }
    )
    obs_holdout = xr.Dataset({"y_holdout": ("time_holdout", RNG.normal(size=N_HOLDOUT))})

    combined_obs = xr.merge([obs, obs_holdout])

    return xr.DataTree.from_dict(
        {
            "observed_data": combined_obs,
            "posterior_predictive": pp,
        }
    )



# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh", "plotly", "none"])
class TestPlotTs:
    """Test suite for plot_ts batteries-included function."""

    def test_plot_ts_basic(self, ts_datatree, backend):
        """plot_ts returns a PlotCollection with basic observed line."""
        pc = plot_ts(ts_datatree, y="y", backend=backend)
        assert isinstance(pc, PlotCollection)
        assert "figure" in pc.viz.data_vars
        assert "observed_line" in pc.viz.children

    def test_plot_ts_with_posterior_predictive(self, ts_datatree, backend):
        """plot_ts draws posterior predictive sample lines when y_hat is given."""
        pc = plot_ts(
            ts_datatree,
            y="y",
            y_hat="y_hat",
            num_samples=10,
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "observed_line" in pc.viz.children
        assert "posterior_predictive" in pc.viz.children

    def test_plot_ts_with_holdout(self, ts_datatree, backend):
        """plot_ts shows holdout scatter and vline when y_holdout is provided."""
        pc = plot_ts(
            ts_datatree,
            y="y",
            y_holdout="y_holdout",
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "observed_line" in pc.viz.children
        assert "observed_scatter" in pc.viz.children
        assert "vline" in pc.viz.children

    def test_plot_ts_with_forecasts(self, ts_datatree, backend):
        """plot_ts draws forecast lines when y_forecasts is given."""
        pc = plot_ts(
            ts_datatree,
            y="y",
            y_holdout="y_holdout",
            y_forecasts="y_forecasts",
            num_samples=10,
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "forecast" in pc.viz.children
        assert "vline" in pc.viz.children

    def test_plot_ts_full(self, ts_datatree, backend):
        """plot_ts works with all visuals enabled together."""
        pc = plot_ts(
            ts_datatree,
            y="y",
            y_hat="y_hat",
            y_holdout="y_holdout",
            y_forecasts="y_forecasts",
            num_samples=5,
            backend=backend,
        )
        assert isinstance(pc, PlotCollection)
        assert "observed_line" in pc.viz.children
        assert "posterior_predictive" in pc.viz.children
        assert "observed_scatter" in pc.viz.children
        assert "forecast" in pc.viz.children
        assert "vline" in pc.viz.children

    def test_plot_ts_disable_visual(self, ts_datatree, backend):
        """Setting a visual to False disables it."""
        pc = plot_ts(
            ts_datatree,
            y="y",
            backend=backend,
            visuals={"observed_line": False, "xlabel": False, "ylabel": False},
        )
        assert isinstance(pc, PlotCollection)
        assert "observed_line" not in pc.viz.children

    def test_plot_ts_invalid_plot_dim(self, ts_datatree, backend):
        """plot_ts raises ValueError for a non-existent plot_dim."""
        with pytest.raises(ValueError, match="not present in the observed data"):
            plot_ts(
                ts_datatree,
                y="y",
                plot_dim="nonexistent_dim",
                backend=backend,
            )
