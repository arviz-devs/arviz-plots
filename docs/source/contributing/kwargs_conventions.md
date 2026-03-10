(kwargs_conventions)=
# Conventions for `visuals`, `aes_by_visuals`, and `stats`

This document describes the naming and usage conventions for the `visuals`, `aes_by_visuals`,
and `stats` parameters in batteries-included plotting functions.

These parameters are used across existing plotting functions and this document describes their current usage patterns. For the full guide on adding a new plot, see {ref}`add_new_plot`.

## Naming conventions

- Use **full names** rather than abbreviations or acronyms
  (e.g., `credible_interval` not `ci`, `point_estimate` not `pe`).
- Each visual element should have a **descriptive name** that clearly indicates
  what it represents (e.g., `trunk`, `twig`, `dist`, `rug`, `title`).

## `visuals`

The `visuals` parameter is a mapping of `{str : mapping or bool}`.
Each key identifies a visual element of the plot, and the value is either:

- A **dict** of keyword arguments passed to the corresponding drawing function.
- `False` to **skip** that visual element entirely.
- `True` or an **empty dict** to use default arguments.

Every visual element has a name which is a valid key in `visuals`.
This allows users to deactivate any operation the plot performs.

Use {func}`~arviz_plots.plots.utils.get_visual_kwargs` to retrieve
the kwargs for a given visual from the `visuals` dict:

```python
from arviz_plots.plots.utils import get_visual_kwargs

density_kwargs = get_visual_kwargs(visuals, "dist")
if density_kwargs is not False:
    # proceed to draw the visual
    ...
```

Some visual elements default to `False` (disabled) and must be explicitly
enabled by the user (e.g., `rug` and `face` in `plot_dist`).

The `remove_axis` key is special: it does not correspond to a drawing function
and can only be set to `False` to skip calling {func}`~arviz_plots.visuals.remove_axis`.

## `aes_by_visuals`

The `aes_by_visuals` parameter is a mapping of `{str : sequence of str or False}`.
It controls which aesthetic mappings from the {class}`~arviz_plots.PlotCollection`
are applied to each visual element.

- Keys in `aes_by_visuals` should **match keys in `visuals`** whenever possible.
- Some keys may only make sense for `visuals` and can be omitted from `aes_by_visuals`
  (e.g., `remove_axis`).
- In rare cases, different visual elements that share the same aesthetic mappings
  and call the same backend function may share a key. This should be done sparingly.

Defaults for `aes_by_visuals` are set using `setdefault` after checking
whether a `plot_collection` has been provided:

```python
if aes_by_visuals is None:
    aes_by_visuals = {}
else:
    aes_by_visuals = aes_by_visuals.copy()
aes_by_visuals.setdefault("dist", plot_collection.aes_set)
```

Use {func}`~arviz_plots.plots.utils.filter_aes` to split the aesthetics for a
specific visual and obtain the dimensions to reduce:

```python
from arviz_plots.plots.utils import filter_aes

reduce_dims, aes, ignore = filter_aes(
    plot_collection, aes_by_visuals, "dist", sample_dims
)
```

## `stats`

The `stats` parameter is a mapping of `{str : mapping or Dataset}`.
Each key identifies a computation, and the value is either:

- A **dict** of keyword arguments passed to the corresponding function in `arviz-stats`.
- A pre-computed {class}`~xarray.Dataset`, interpreted as already-computed values
  for that statistic.

There are no strict naming rules for `stats` keys yet. In existing plots, the keys
generally match the name of the visual they feed into
(e.g., `"dist"`, `"credible_interval"`, `"point_estimate"`,
`"trunk"`, `"twig"`).

## Summary of conventions

| Aspect | Convention |
|---|---|
| Naming | Full names, no abbreviations |
| `visuals` keys | One per visual element; set to `False` to disable |
| `aes_by_visuals` keys | Match `visuals` keys where applicable |
| `stats` keys | Generally match the visual they compute data for |
| Mutability | Always copy mutable inputs (`visuals.copy()`, etc.) |
| Defaults | Use `setdefault` for kwargs; check active aesthetics before setting defaults |
