"""Batteries-included ArviZ plots."""

from arviz_plots.plots.autocorr_plot import plot_autocorr
from arviz_plots.plots.bf_plot import plot_bf
from arviz_plots.plots.combine import combine_plots
from arviz_plots.plots.compare_plot import plot_compare
from arviz_plots.plots.convergence_dist_plot import plot_convergence_dist
from arviz_plots.plots.dgof_dist_plot import plot_dgof_dist
from arviz_plots.plots.dgof_plot import plot_dgof
from arviz_plots.plots.dist_plot import plot_dist
from arviz_plots.plots.ecdf_plot import plot_ecdf_pit
from arviz_plots.plots.energy_plot import plot_energy
from arviz_plots.plots.ess_plot import plot_ess
from arviz_plots.plots.evolution_plot import plot_ess_evolution
from arviz_plots.plots.forest_plot import plot_forest
from arviz_plots.plots.khat_plot import plot_khat
from arviz_plots.plots.lm_plot import plot_lm
from arviz_plots.plots.loo_interval_plot import plot_loo_interval
from arviz_plots.plots.loo_pava_plot import plot_loo_pava
from arviz_plots.plots.loo_pit_plot import plot_loo_pit
from arviz_plots.plots.mcse_plot import plot_mcse
from arviz_plots.plots.pair_focus_plot import plot_pair_focus
from arviz_plots.plots.pair_plot import plot_pair
from arviz_plots.plots.parallel_plot import plot_parallel
from arviz_plots.plots.pava_calibration_plot import plot_ppc_pava
from arviz_plots.plots.pava_residual_plot import plot_ppc_pava_residuals
from arviz_plots.plots.ppc_censored_plot import plot_ppc_censored
from arviz_plots.plots.ppc_dist_pit_plot import plot_ppc_dist_pit
from arviz_plots.plots.ppc_dist_plot import plot_ppc_dist
from arviz_plots.plots.ppc_interval_plot import plot_ppc_interval
from arviz_plots.plots.ppc_pit_plot import plot_ppc_pit
from arviz_plots.plots.ppc_rootogram_plot import plot_ppc_rootogram
from arviz_plots.plots.ppc_tstat import plot_ppc_tstat
from arviz_plots.plots.prior_posterior_plot import plot_prior_posterior
from arviz_plots.plots.psense_dist_plot import plot_psense_dist
from arviz_plots.plots.psense_quantities_plot import plot_psense_quantities
from arviz_plots.plots.rank_dist_plot import plot_rank_dist
from arviz_plots.plots.rank_plot import plot_rank
from arviz_plots.plots.ridge_plot import plot_ridge
from arviz_plots.plots.trace_dist_plot import plot_trace_dist
from arviz_plots.plots.trace_plot import plot_trace
from arviz_plots.plots.utils import add_bands, add_lines

__all__ = [
    "combine_plots",
    "plot_autocorr",
    "plot_bf",
    "plot_compare",
    "plot_convergence_dist",
    "plot_dgof",
    "plot_dgof_dist",
    "plot_dist",
    "plot_forest",
    "plot_loo_interval",
    "plot_loo_pava",
    "plot_trace",
    "plot_trace_dist",
    "plot_ecdf_pit",
    "plot_energy",
    "plot_ess",
    "plot_ess_evolution",
    "plot_khat",
    "plot_loo_pit",
    "plot_mcse",
    "plot_pair",
    "plot_pair_focus",
    "plot_parallel",
    "plot_ppc_censored",
    "plot_ppc_dist",
    "plot_ppc_dist_pit",
    "plot_ppc_interval",
    "plot_ppc_rootogram",
    "plot_prior_posterior",
    "plot_rank",
    "plot_rank_dist",
    "plot_ridge",
    "plot_ppc_pava",
    "plot_ppc_pava_residuals",
    "plot_ppc_pit",
    "plot_ppc_tstat",
    "plot_psense_dist",
    "plot_psense_quantities",
    "add_lines",
    "add_bands",
    "plot_lm",
]
