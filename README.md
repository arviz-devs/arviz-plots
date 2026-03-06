# arviz-plots

[![Run tests](https://github.com/arviz-devs/arviz-plots/actions/workflows/test.yml/badge.svg)](https://github.com/arviz-devs/arviz-plots/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/arviz-devs/arviz-plots/graph/badge.svg?token=1VIPLXCOJQ)](https://codecov.io/gh/arviz-devs/arviz-plots)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)

ArviZ (pronounced "AR-_vees_") is a Python package for exploratory analysis of Bayesian models. It includes functions for posterior analysis, data storage, model checking, comparison and diagnostics.

arviz-plots is the subpackage in charge of the visualizations.

### ArviZ in other languages
ArviZ also has a Julia wrapper available [ArviZ.jl](https://julia.arviz.org/).

## Documentation

The ArviZ documentation can be found in the [official docs](https://python.arviz.org).
Here are some quick links for common scenarios:

* First time Bayesian modelers and ArviZ users: [EABM book](https://arviz-devs.github.io/EABM/)
* First time ArviZ users, already familiar with Bayesian modeling: [overview notebook](https://python.arviz.org/projects/plots/en/latest/tutorials/overview.html) or [example gallery](https://python.arviz.org/projects/plots/en/latest/gallery/index.html)
* ArviZ 0.x user: [migration guide](https://python.arviz.org/en/latest/user_guide/migration_guide.html)
* ArviZ-verse documentation:
  - [arviz-base](https://python.arviz.org/projects/base/en/latest/)
  - [arviz-stats](https://python.arviz.org/projects/stats/en/latest/)
  - [arviz-plots](https://python.arviz.org/projects/plots/en/latest/) (this package)


## Installation

### Stable
ArviZ is available for installation from [PyPI](https://pypi.org/project/arviz/).
The latest stable version can be installed using pip:

```
pip install "arviz-plots"
```

ArviZ is also available through [conda-forge](https://anaconda.org/conda-forge/arviz).

```
conda install -c conda-forge arviz arviz-plots
```

### Development
The latest development version can be installed from the main branch using pip:

```
pip install git+git://github.com/arviz-devs/arviz-plots.git
```

Another option is to clone the repository and install using git and setuptools:

```
git clone https://github.com/arviz-devs/arviz-plots.git
cd arviz
python setup.py install
```

## Citation


If you use ArviZ and want to cite it please use [![DOI](https://joss.theoj.org/papers/10.21105/joss.09889/status.svg)](https://doi.org/10.21105/joss.09889)

Here is the citation in BibTeX format

```
@article{Martin2026,
doi = {10.21105/joss.09889},
url = {https://doi.org/10.21105/joss.09889},
year = {2026},
publisher = {The Open Journal},
volume = {11},
number = {119},
pages = {9889},
author = {Martin, Osvaldo A. and Abril-Pla, Oriol and Deklerk, Jordan and Axen, Seth D. and Carroll, Colin and Hartikainen, Ari and Vehtari, Aki},
title = {ArviZ: a modular and flexible library for exploratory analysis of Bayesian models},
journal = {Journal of Open Source Software}}
```


## Contributions
ArviZ is a community project and welcomes contributions.
Additional information can be found in the [contributing guide](https://python.arviz.org/en/latest/contributing/index.html)


## Code of Conduct
ArviZ wishes to maintain a positive community. Additional details
can be found in the [Code of Conduct](https://www.arviz.org/en/latest/CODE_OF_CONDUCT.html)

## Donations
ArviZ is a non-profit project under NumFOCUS umbrella. If you want to support ArviZ financially, you can donate [here](https://numfocus.org/donate-to-arviz).

## Sponsors and Institutional Partners
[![Aalto University](https://raw.githubusercontent.com/arviz-devs/arviz-project/refs/heads/main/cards/Aalto-black-text.png)](https://www.aalto.fi/en)
[![FCAI](https://raw.githubusercontent.com/arviz-devs/arviz-project/refs/heads/main/cards/FCAI.png)](https://fcai.fi/)
[![NumFOCUS](https://raw.githubusercontent.com/arviz-devs/arviz-project/refs/heads/main/sphinx/NumFocus.png)](https://numfocus.org)

[The ArviZ project website](https://www.arviz.org/en/latest/sponsors_partners.html) has more information about each sponsor and the support they provide.
