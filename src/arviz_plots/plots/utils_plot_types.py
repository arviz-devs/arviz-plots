"""Utility functions to check data types for plotting functions."""
import warnings

import numpy as np


def warn_if_binary(observed_dist, predictive_dist):
    """Warn if data is binary."""
    for dist, name in zip([observed_dist, predictive_dist], ["observed", "predictive"]):
        if dist is None:
            continue
        if any(set(np.unique(dist[var].values)).issubset({0, 1}) for var in dist.data_vars):
            warnings.warn(
                f"The {name} data looks binary. For binary outcomes, "
                "plot_ppc_pava may be more appropriate.",
                stacklevel=2,
            )


def warn_if_discrete(observed_dist, predictive_dist, kind):
    """Warn if data is discrete."""
    observed_discrete, predictive_discrete = get_discrete_flags(observed_dist, predictive_dist)

    if any(predictive_discrete + observed_discrete) and kind != "ecdf":
        warnings.warn(
            "Detected at least one discrete variable.\n"
            "Consider using plot_ppc variants specific for discrete data, "
            "such as plot_ppc_pava or plot_ppc_rootogram.",
            UserWarning,
            stacklevel=2,
        )


def raise_if_continuous(observed_dist, predictive_dist):
    """Raise error if data is continuous."""
    observed_discrete, predictive_discrete = get_discrete_flags(observed_dist, predictive_dist)
    if not all(predictive_discrete + observed_discrete):
        raise ValueError(
            "Detected at least one continuous variable.\n"
            "This function only works for discrete data.\n"
            "Consider using other functions such as plot_ppc_dist\n"
            "plot_ppc_pit, or plot_ppc_tstat.",
        )


def warn_if_prior_predictive(group):
    """Warn if group is prior_predictive."""
    if group == "prior_predictive":
        warnings.warn(
            "\n`This plot always use the `observed_data` group."
            "\nBe cautious when using it for prior predictive checks.",
            UserWarning,
            stacklevel=2,
        )


def get_discrete_flags(observed_dist, predictive_dist):
    """Get list of discrete flags for observed and predictive distributions."""
    predictive_discrete = [
        predictive_dist[var].values.dtype.kind == "i" for var in predictive_dist.data_vars
    ]

    if observed_dist is not None:
        observed_discrete = [
            observed_dist[var].values.dtype.kind == "i" for var in observed_dist.data_vars
        ]
    else:
        observed_discrete = []

    return observed_discrete, predictive_discrete
