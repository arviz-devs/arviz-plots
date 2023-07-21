# arviz-base
ArviZ base features and converters.

## Installation

It currenly can only be installed with pip and from GitHub:

```bash
pip install arviz-base @ git+https://github.com/arviz-devs/arviz-base
```

Note that `arviz-base` is a minimal package, which only depends on
xarray (and xarray-datatree), numpy and typing-extensions.
Everything else (netcdf, zarr, dask...) are optional dependencies.
This allows installing only those that are needed, e.g. if you
only plan to use zarr, there is no need to install netcdf.

For convenience, some bundles are available to be installed with:

```bash
pip install "arviz-base[<option>] @ git+https://github.com/arviz-devs/arviz-base"
```

where `<option>` can be one of:

* `netcdf`
* `h5netcdf`
* `zarr`
* `test` (for developers)
* `doc` (for developers)


You can install multiple bundles of optional dependencies separating them with commas.
Thus, to install all user facing optional dependencies you should use `xarray-einstats[einops,numba]`

```{toctree}
:hidden:

api/index
```

```{toctree}
:caption: About
:hidden:

Twitter <https://twitter.com/arviz_devs>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/xarray-einstats>
```
