"""Utility functions to check data types for plotting functions."""

import warnings

import numpy as np


def warn_if_binary(observed_dist, predictive_dist):
    """Warn if data is binary."""
    for dist, name in zip([observed_dist, predictive_dist], ["observed_data", "predictive"]):
        if dist is None:
            continue
        binary_vars = [
            var for var, da in dist.items() if (np.isclose(da, 0) | np.isclose(da, 1)).all()
        ]
        if binary_vars:
            warnings.warn(
                f"Variables {', '.join(binary_vars)} in '{name}' look binary. "
                "For binary outcomes, plot_ppc_pava may be more appropriate.",
                UserWarning,
                stacklevel=2,
            )


def warn_if_discrete(observed_dist, predictive_dist, kind):
    """Warn if data is discrete."""
    for dist, name in zip([observed_dist, predictive_dist], ["observed", "predictive"]):
        discrete_flags = get_discrete_flags(dist)
        discrete_vars = [name for name, flag in discrete_flags.items() if flag]
        if discrete_vars and kind != "ecdf":
            warnings.warn(
                f"Variables {', '.join(discrete_vars)} in '{name}' look discrete.\n"
                "Consider using plot_ppc variants specific for discrete data, "
                "such as plot_ppc_pava or plot_ppc_rootogram.",
                UserWarning,
                stacklevel=2,
            )


def raise_if_continuous(observed_dist, predictive_dist):
    """Raise error if data is continuous."""
    for dist, name in zip([observed_dist, predictive_dist], ["observed", "predictive"]):
        discrete_flags = get_discrete_flags(dist)
        continuous_vars = [name for name, flag in discrete_flags.items() if not flag]
        if continuous_vars:
            raise ValueError(
                f"Variables {', '.join(continuous_vars)} in '{name}' are continuous.\n"
                "This function only works for discrete data.\n"
                "Consider using other functions such as plot_ppc_dist\n"
                "plot_ppc_pit, or plot_ppc_tstat.",
            )


def warn_if_prior_predictive(group):
    """Warn if group is prior_predictive."""
    if group == "prior_predictive":
        warnings.warn(
            "This plot always uses the `observed_data` group."
            "\nBe cautious when using it for prior predictive checks.",
            UserWarning,
            stacklevel=2,
        )


def get_discrete_flags(group):
    """Get a name-flag mapping to indicate which variables are discrete."""
    flags = {}
    if group is not None:
        for var in group.data_vars:
            flags[var] = group[var].values.dtype.kind == "i"
    return flags
