"""Rank plot visualization.

Rank plots show the distribution of posterior draws ranked across chains,
which helps assess convergence.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_rank(data, var_names=None, bins=20, kind="bars", colors="cycle", figsize=(12, 10)):
    """
    Create rank plots to assess MCMC convergence across chains.

    Rank plots display histograms of the ranked posterior draws (ranked over all chains)
    plotted separately for each chain. In well-mixed chains targeting the same posterior,
    the ranks in each chain should be uniformly distributed. Deviations from uniformity
    indicate potential convergence problems, differences in location/scale parameters,
    or poor mixing between chains.

    This plot was introduced by Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021)
    in "Rank-normalization, folding, and localization: An improved R-hat for assessing
    convergence of MCMC" (Bayesian Analysis).

    Parameters
    ----------
    data : arviz InferenceData object or compatible
        The ArviZ inference data containing posterior samples
    var_names : str or list, optional
        Variables to plot. If None, all variables are plotted
    bins : int, optional
        Number of bins for histogram. Default is 20
    kind : {"bars", "vlines"}, default "bars"
        If "bars", ranks are displayed as stacked histograms (one per chain)
        If "vlines", ranks are displayed as vertical lines with reference line at 0
    colors : str or list, optional
        Colors for chains. If "cycle", uses matplotlib's color cycle
        If a single color string, uses that color for all chains
        If a list, uses each color for corresponding chain
    figsize : tuple, optional
        Figure size in inches. Default is (12, 10)

    Returns
    -------
    fig, axes : matplotlib figure and axes
        The figure and array of axes containing the rank plots

    Notes
    -----
    This implementation specifically handles multidimensional parameters by:
    1. Creating separate plots for each dimension of multivariate parameters
    2. Preserving parameter names and coordinates where applicable
    3. For the 'theta' parameter with 'school' dimension, each school gets its own plot

    Examples
    --------
    Basic usage with default settings:

    >>> import arviz as az
    >>> import matplotlib.pyplot as plt
    >>> data = az.load_arviz_data('centered_eight')
    >>> fig, axes = plot_rank(data)
    >>> plt.show()

    Compare different variables with custom settings:

    >>> fig, axes = plot_rank(data, var_names=['mu', 'tau'],
    ...                       kind='vlines', bins=30,
    ...                       colors=['blue', 'green'],
    ...                       figsize=(10, 6))
    >>> plt.show()

    Plot specific dimensions of multidimensional parameter:

    >>> fig, axes = plot_rank(data, var_names='theta')  # plots each school separately
    >>> plt.show()

    Comparing convergence across different models:

    >>> centered = az.load_arviz_data('centered_eight')
    >>> noncentered = az.load_arviz_data('non_centered_eight')
    >>>
    >>> fig1, axes1 = plot_rank(centered, var_names='mu')
    >>> plt.title('Centered Model')
    >>> plt.show()
    >>>
    >>> fig2, axes2 = plot_rank(noncentered, var_names='mu')
    >>> plt.title('Non-centered Model')
    >>> plt.show()

    Interpreting Results:
    ---------------------
    - Uniform rank distributions across chains suggest good mixing and convergence
    - U-shaped distributions suggest overdispersion (chains exploring different regions)
    - Inverted-U shapes suggest underdispersion (chains not fully exploring the space)
    - One chain consistently higher/lower than others suggests poor convergence
    """
    # Get posterior data
    if hasattr(data, "posterior"):
        posterior = data.posterior
    else:
        posterior = data

    # Get variable names
    if var_names is None:
        var_names = list(posterior.data_vars)
    elif isinstance(var_names, str):
        var_names = [var_names]

    print(f"Available variables: {var_names}")
    for var in var_names:
        if var in posterior.data_vars:
            print(f"Variable {var} has dimensions: {posterior[var].dims}")

    # PRE-CALCULATION: Count total number of plots needed BEFORE creating figure
    total_plots = 0
    plot_info = []  # Will store (var_name, data, title) for each plot

    for var_name in var_names:
        if var_name not in posterior.data_vars:
            continue

        var = posterior[var_name]

        # Handle multidimensional variables
        if len(var.dims) > 2:
            # For theta with school dimension, create separate plot for each school
            if var_name == "theta" and "school" in var.dims:
                for school_name in var.coords["school"].values:
                    sel_data = var.sel(school=school_name)

                    # Ensure we have chain and draw dimensions in right order
                    if "chain" in sel_data.dims and "draw" in sel_data.dims:
                        chain_idx = sel_data.dims.index("chain")
                        draw_idx = sel_data.dims.index("draw")
                        data_arr = sel_data.values

                        # Transpose if needed to get (chain, draw) order
                        if chain_idx > draw_idx:
                            data_arr = data_arr.T

                        plot_info.append((var_name, data_arr, f"{var_name} {school_name}"))
                        total_plots += 1
            else:
                # For other multidimensional variables, just plot first element
                extra_dims = [d for d in var.dims if d not in ["chain", "draw"]]
                if extra_dims:
                    idx = {d: 0 for d in extra_dims}
                    var = var.isel(**idx)
                    idx_str = ",".join([str(0) for _ in range(len(extra_dims))])
                    title = f"{var_name}[{idx_str}]"
                else:
                    title = var_name

                if "chain" in var.dims and "draw" in var.dims:
                    chain_idx = var.dims.index("chain")
                    draw_idx = var.dims.index("draw")
                    data_arr = var.values

                    # Transpose if needed to get (chain, draw) order
                    if chain_idx > draw_idx:
                        data_arr = data_arr.T

                    plot_info.append((var_name, data_arr, title))
                    total_plots += 1
        else:
            # Simple 2D case with chain and draw
            if "chain" in var.dims and "draw" in var.dims:
                chain_idx = var.dims.index("chain")
                draw_idx = var.dims.index("draw")
                data_arr = var.values

                # Transpose if needed to get (chain, draw) order
                if chain_idx > draw_idx:
                    data_arr = data_arr.T

                plot_info.append((var_name, data_arr, var_name))
                total_plots += 1

    print(f"Total plots to be created: {total_plots}")
    print(f"Plot titles: {[info[2] for info in plot_info]}")

    # Now create the correct number of subplots
    n_cols = min(5, total_plots)
    n_rows = int(np.ceil(total_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Set up colors
    n_chains = len(posterior.chain)
    if colors == "cycle":
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])[:n_chains]
    elif isinstance(colors, str):
        colors = [colors] * n_chains

    # Plot each variable
    for i, (var_name, var_data, title) in enumerate(plot_info):
        if i >= len(axes):
            print(f"Warning: Not enough axes for {title} (index {i}, axes length {len(axes)})")
            break

        ax = axes[i]

        # Calculate ranks
        n_chains, n_draws = var_data.shape
        data_all = var_data.flatten()
        sort_idx = np.argsort(data_all)
        ranks = np.zeros_like(sort_idx)
        ranks[sort_idx] = np.arange(len(data_all))
        ranks = ranks.reshape(n_chains, n_draws)

        # Plot according to kind
        if kind == "bars":
            for c in range(n_chains):
                ax.hist(
                    ranks[c],
                    bins=bins,
                    alpha=0.6,
                    color=colors[c % len(colors)],
                    label=f"Chain {c}",
                    density=True,
                )

            # Reference line for uniform distribution
            ax.axhline(1.0 / len(data_all), color="k", linestyle="--", alpha=0.6)

        elif kind == "vlines":
            for c in range(n_chains):
                chain_ranks = ranks[c]
                positions = np.arange(len(chain_ranks))
                heights = chain_ranks - (len(data_all) / 2)
                heights_normalized = heights / (len(data_all) / 2)  # Scale to [-1, 1]

                ax.vlines(
                    positions,
                    np.zeros_like(positions),
                    heights_normalized,
                    color=colors[c % len(colors)],
                    alpha=0.6,
                )
                ax.plot(
                    positions,
                    heights_normalized,
                    "o",
                    color=colors[c % len(colors)],
                    alpha=0.6,
                    label=f"Chain {c}",
                )

            # Reference line at zero
            ax.axhline(0, color="k", linestyle="--", alpha=0.6)

        # Set labels and title
        if kind == "bars":
            ax.set_xlabel("Rank")
            ax.set_ylabel("Frequency")
        else:
            ax.set_xlabel("Draw")
            ax.set_ylabel("Relative Position")

        ax.set_title(title)

        # Add legend but keep it small and out of the way
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize="x-small", loc="best")

    # Hide unused axes
    for j in range(len(plot_info), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig, axes
