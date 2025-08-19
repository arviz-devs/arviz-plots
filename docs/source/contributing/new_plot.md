(add_new_plot)=
# Adding a batteries-included plot
Each batteries-included plot should have its own file within the `/src/arviz-plots/plots` folder.

:::{important}
Batteries-included plots provide opinionated defaults for common EABM tasks.
Consequently, setting those defaults should represent a significant part of both
the code itself and the work done.

In fact, there are cases where getting the function to work is easier than
managing to generate sensible defaults for the different parameter combinations.
:::

## Call signature
All functions in {ref}`plots_api` should have a similar
signature.

This includes things like taking {class}`.DataTree` (in some cases also dictionaries of `DataTrees`) as first input,
having inputs for defining subsets with `var_names` or `coords`,
for the dimensions to reduce...

Also have inputs that are passed downstream to customize objects created when plotting like `visuals` or `pc_kwargs`.

And last but not least, they should all return a {class}`~arviz_plots.PlotCollection`.

:::{dropdown} Function signature and docstring template
:name: new_plot/common_signature

Here is a template of the function signature with common default values as well as their signature. 

There are multiple parts within the template that require extra imput that is plot dependent.
These placeholders are indicated with `[[description on what to fill here]]`

```python
def plot_xyz(
    # initial base arguments
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    # plot specific arguments
    [[...]],
    # more base arguments
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals=None,
    visuals=None,
    stats=None,
    **pc_kwargs,
):
    
    """Plot description in 1 line.

    Extended description.

    Parameters
    ----------
    [[choose one]]
    dt : DataTree
    dt : DataTree or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to aesthetics.
    var_names : str or sequence of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, “like”, “regex”}, default None
        If None, interpret `var_names` as the real variables names.
        If “like”, interpret `var_names` as substrings of the real variables names.
        If “regex”, interpret `var_names` as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce unless mapped to an aesthetic.
        Defaults to ``rcParams["data.sample_dims"]``
    [[...]]
    plot_collection : PlotCollection, optional
    backend : {"matplotlib", "bokeh", "plotly", "none"}, optional
        Plotting backend to use. Defaults to ``rcParams["plot.backend"]``
    labeller : labeller, optional
    aes_by_visuals : mapping of {str : sequence of str or False}, optional
        Mapping of visuals to aesthetics that should use their mapping in `plot_collection`
        when plotted. Valid keys are the same as for `visuals`.

        [[Description of default aesthetic mappings]]
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * [[first visual id]] -> [[function called when drawing the first visual]]
        * [[repeat for all visuals]]

    stats : mapping, optional
        Valid keys are:

        * [[stats/summary/diagnostic name]] -> [[function in arviz-stats used for computation]]
        * [[repeat for all computations]]

    pc_kwargs : mapping
        [[choose one]]
        Passed to :class:`arviz_plots.PlotCollection.wrap`
        Passed to :class:`arviz_plots.PlotCollection.grid`

    Returns
    -------
    PlotCollection
    """
```

:::

## Initial defaults
The first thing that should generally happen in a `plot_xyz` function is processing
the input arguments and setting initial defaults. This generally means creating
mutable objects for those input arguments the function expects to be mutable objects (e.g. dicts)
and getting default values from rcParams.

The general templates are therefore:

```python
# rcParams
if parameter is None:
    parameter = rcParams["xyz.parameter"]

# mutable inputs
if xyz_kwargs is None:
    xyz_kwargs = {}
# if xyz is modified by the function, then also add
else:
    xyz_kwargs = xyz_kwargs.copy()
```

There are also cases where this includes converting multiple input types into the
type of object actually used within the function.

One example is `sample_dims` which can be a string or a sequence, but the functions
expect to be a sequence. Thus, if it's `None` we use the above template to get the
default value, then if it is a string we create a list with the string as only element of the list:

```python
if sample_dims is None:
    sample_dims = rcParams["data.sample_dims"]
if isinstance(sample_dims, str):
    sample_dims = [sample_dims]
```

Another example is processing `dt`, `var_names`, `group`... where several arguments
can take multiple types and values. In this case, there is a helper function to take
care of that:

```
distribution = process_group_variables_coords(
    dt, group=group, var_names=var_names, filter_vars=filter_vars, coords=coords
)
```

Finally, _if needed_ we set default arguments for {class}`~arviz_plots.PlotCollection`
and create and instance of it. In addition to having to copy `pc_kwargs` when it is
not `None` following the template above, it might also be necessary to set defaults
for dictionaries within `pc_kwargs` such as `pc_kwargs["aes"]`.

When creating a `PlotCollection` from scratch, the :term:`figure` size should also be set
as a function of the grid size that will be needed. `arviz_plots.plots.utils` has two
functions to this end, depending on wheather `.grid` or `.wrap` methods will be called:
`set_wrap_layout` and `set_grid_layout`.

Use the following pattern for such cases:

```python
pc_kwargs["aes"] = pc_kwargs.get("aes", {}).copy()
pc_kwargs["aes"].setdefault("color", ["model"])
[...]
pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, ds)
```

## PlotCollection dependent defaults
Once we have made sure that `plot_collection` is not None, we continue setting defaults.
There are arguments such as `aes_by_visuals` that need information from the `plot_collection`
input to have their defaults set. A common pattern will therefore be:

```python
if aes_by_visuals is None:
    aes_by_visuals = {}
else:
    aes_by_visuals = aes_by_visuals.copy()
aes_by_visuals.setdefault("visual", plot_collection.aes_set)
aes_by_visuals.setdefault("annotation", ["color"])
```

where we are setting the default that "visual" will use all available aesthetic mappings,
and "annotation" will use only the mapping for color (if set, that is checked later on, so this default can be hardcoded).

We might also want to tweak some aesthetic values, in which case
{meth}`~.PlotCollection.get_aes_as_dataset` and {meth}`~.PlotCollection.update_aes_from_dataset` can be helpful. See {func}`~arviz_plots.plot_forest` source code for an example.

## Adding {term}`visuals` to the {term}`plot`

:::{important}
Before starting to add visuals individually, check if part of the plot can be composed
calling an existing function.

For example, {func}`~arviz_plots.plot_trace_dist` calls {func}`~arviz_plots.plot_trace`
to fill the right column and {func}`~arviz_plots.plot_dist` to fill the left one.
:::

Each visual should have its own id and ideally also its own call to `.map`.
The id is what is used to get the visual specific kwargs from `visuals` and `aes_by_visuals`,
and what is used to store the visual in the {attr}`~.PlotCollection.viz` attribute.
The independent calls to `.map` allow each visual in the plot to use different {term}`aesthetic mappings`.
Consequently, there are multiple steps that should be followed for each visual:

1. Access the respective kwargs in `visuals`. Only proceed to step 2 if these are not `False`.
2. Use `filter_aes` to get the dimensions, active aesthetics and aesthetics to be ignored for this
   particular visual.
3. (optional) If necessary and particular to this visual, call the stats/summary/diagnostic
   function. Details on this are in the {ref}`new_plot/computation` section.
   This will often have already happened earlier on in which case these steps might not happen
   consecutively within the code.
4. Set default arguments.
   - Check we aren't overriding active aesthetics by setting defaults.
   - Use only properties that are part of the {ref}`common interface <backend_interface_arguments>` for defaults.
     If a setting a default for a property not on the list, open an issue to discuss it.
     This is key to ensure all plotting backends work seamlessly and behave as expected.
5. Call {meth}`~.PlotCollection.map` to create the visual. For some "visuals" like removing
   axis, it doesn't make much sense to store them, in such cases, use `store_artist=backend == "none"`.

Here is a general template:

```python
# step 1
artist_kwargs = copy(visuals.get("visual", {}))
if artist_kwargs is not False:
    # step 2
    artist_dims, artist_aes, artist_ignore = filter_aes(
        plot_collection, aes_by_visuals, "visual", sample_dims
    )
    # step 3 (optional)
    artist_data = stats(..., dims=artist_dims, **stats.get("visual", {}))

    # step 4
    if "color" not in artist_aes:
        artist_kwargs.setdefault("color", "gray")

    # step 5
    plot_collection.map(
        visual,
        "visual",
        data=artist_data, # optional
        ignore_aes=artist_ignore,
        ..., # if needed, add more arguments
        **artist_kwargs
    )
```

:::{note}
There might be some cases in which multiple visuals are given the same aesthetic mappings
and keyword arguments, but this should be done rarely and for visuals that eventually call
the same function in the {mod}`~arviz_plots.backend` module.

One example of this particular case are the var+coord labels in `plot_trace_dist`.
The left column labels the x axis with the variable name and coordinate subset
whereas the right column labels the y axis.

Therefore, there are to `.map` calls,
one to {func}`~.visuals.labelled_x` and {func}`~.visuals.labelled_y` but they can
be considered the same element, so they both get `visuals` and `aes_by_visuals` from
the `label` kwarg.
:::

(new_plot/computation)=
## Computation
:::{warning}
Computation relies on `arviz-stats` library, which is still in earlier development stages
than arviz-plots. So at this point there aren't many recommendations on the functions themselves
:::

Functions should reduce the dimensions returned by `filter_aes` (`artist_dims` above).
Moreover, in order for the result to be valid `data` argument when calling `.map` it must
be a `Dataset` with the same variables in `var_names` (or all variables in input data if not given).
