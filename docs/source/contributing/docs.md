# Documentation

## How to build the documentation locally
Similarly to testing, there are also tox jobs that take care of building the right environment
and running the required commands to build the documentation.

In general the process should follow these three steps in this order:

```console
tox -e cleandocs
tox -e docs # or tox -e nogallerydocs
tox -e viewdocs
```

These commands will respectively:

* Delete all intermediate sphinx files that were generated during the build process
* Run `sphinx-build` command to parse and render the library documentation
* Open the documentation homepage on the default browser with `python -m webbrowser`

The only required step however is the middle one. In general sphinx uses the intermediate
files only if it detects it hasn't been modified, so when iterating quickly it is recommended
to skip the clean step in order to achieve faster builds. Moreover, if the documentation
page is already open on the browser, there is no need for the viewdocs job because
the documentation is always rendered on the same path; refreshing the page from the browser
is enough.

The example gallery requires processing the python scripts in order to execute each
once per backend in order to generate the png or html+javascript preview.
Therefore, it is the most time consuming step of generating the documentation.
As very often we'll work on the part of the docs not related to the example gallery,
the command `tox -e nogallerydocs` will generate the documentation without the example gallery,
which allows for much faster iteration when writing documentation.
This also means for example the `minigallery` directive in the docstrings won't work,
and sphinx will output warnings about it when using this option.


## How to add examples to the gallery
Examples in the gallery are written in the form of python scripts.
They are divided between multiple categories,
with each category being a folder within `/docs/source/gallery/`.
Therefore, anything matching this glob `/docs/source/gallery/**/*.py`
will be rendered into the example gallery.
To control the order in which examples appear in the gallery,
all filenames should start with two digits, underscore and then
the unique name of the script.

The script is divided in two parts, the first is a file level docstring,
the second is the code example itself.

The docstring part should contain a markdown top level title,
a short description of the example and a seealso directive using MyST syntax.

The code part should import arviz-plots as `azp`. Later on, it set `backend="none"`
explicitly when calling the plotting functions and
store the generated {class}`~arviz_plots.PlotCollection` as the `pc` variable
so the example can finish with `pc.show()`.

Here is an example that can be used as template:

```python
"""
# Posterior ECDFs

Faceted ECDF plots for 1D marginals of the distribution

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_dist`

EABM chapter on [Visualization of Random Variables with ArviZ](https://arviz-devs.github.io/EABM/Chapters/Distributions.html#distributions-in-arviz)
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")
pc = azp.plot_dist(
    data,
    kind="ecdf",
    pc_kwargs={"col_wrap": 4},
    backend="none"  # change to preferred backend
)
pc.show()
```

## About arviz-plots documentation
Documentation for arviz-plots is written in both rST and MyST (which can be used from jupyter
notebooks too) and rendered with Sphinx. Docstrings follow the numpydoc style guide.

### The gallery generator sphinxext
We have a custom sphinx extension to generate the example gallery, located at
`/docs/sphinxext/gallery_generator.py`.

This sphinx extension reads the example scripts within `/docs/source/gallery`
and takes care of the following tasks:

1. Process the script contents into a MyST source page with proper syntax, tabs, code block,
   and relevant links.
1. Execute the code for all plotting backends to generate the respective previews.
   In addition, when executing the matplotlib version, it is stored as a png to use as the
   miniature in the gallery page.
1. Generate the index page for the gallery, with the grid view
1. Generate a json with references of all the functions used in the different examples.
   This supports the `minigallery` directive that allows adding plot specific galleries
   in the examples section of the docstring.
