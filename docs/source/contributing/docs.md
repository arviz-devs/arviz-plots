# Documentation

## How to build the documentation locally
Similarly to testing, there are also tox jobs that take care of building the right environment
and running the required commands to build the documentation.

In general the process should follow these three steps in this order:

```console
tox -e cleandocs
tox -e docs
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

## About arviz-plots documentation
Documentation for arviz-plots is written in both rST and MyST (which can be used from jupyter
notebooks too) and rendered with Sphinx. Docstrings follow the numpydoc style guide.
