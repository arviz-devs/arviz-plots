# Adding a batteries-included plot
Each batteries-included plot should have its own file within the `/src/arviz-plots/plots` folder.

:::{important}
Batteries-included plots provide opinionated defaults for common EABM tasks.
Consequently, setting those defaults should represent a significant part of both
the code itself and the work done.

In fact, there are cases where getting the function to work is easier than
managing to generate sensible defaults for the different parameter combinations.
:::

## Initial defaults
WIP: setting initial defaults independent of plot_collection, call process_data_var_names_coords,
get relavant rcParams...


## PlotCollection dependent defaults
WIP: generate PlotCollection dependent defaults, aes_map dictionary,
custom aes value modification, 


## Computation
WIP: call functions from arviz-stats to compute summaries and diagnostics


## `.map` usage pattern
WIP: use map once per artist, only if the artist key in plot_kwargs is not False,
add defaults checking against active aes_keys
