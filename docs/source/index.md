# arviz-plots
ArviZ plotting elements and batteries included plots.

## Installation

It currenly can only be installed with pip:

::::{tab-set}
:::{tab-item} PyPI
:sync: stable

```bash
pip install "arviz-plots[<backend>]"
```
:::
:::{tab-item} GitHub
:sync: dev

```bash
pip install "arviz-plots[<backend>] @ git+https://github.com/arviz-devs/arviz-plots"
```
:::
::::

Note that `arviz-plots` is a minimal package, which only depends on
xarray (and xarray-datatree), numpy, arviz-base and arviz-stats.
None of the possible backends: matplotlib, bokeh or plotly is installed
by default.

Consequently, it is not recommended to install `arviz-plots` but
instead to choose which backend to use. For example `arviz-plots[bokeh]`
or `arviz-plots[matplotlib, plotly]`, multiple comma separated values are valid too.

This will ensure all relevant dependencies are installed. For example, to use the plotly backend,
both `plotly>5` and `webcolors` are required.

```{toctree}
:hidden:
:caption: User guide

tutorials/plots_intro
tutorials/intro_to_plotcollection
tutorials/compose_own_plot
```

```{toctree}
:hidden:
:caption: Reference

gallery/index
api/index
glossary
```

```{toctree}
:hidden:
:caption: Contributing

contributing/testing
contributing/new_plot
contributing/docs
```

```{toctree}
:caption: About
:hidden:

Twitter <https://twitter.com/arviz_devs>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/arviz-plots>
```
