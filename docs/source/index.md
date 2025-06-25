# ArviZ-plots

Welcome to the ArviZ-plots documentation! This library focuses on visual summaries and diagnostics for exploratory analysis of Bayesian models. It is one of the 3 components of the ArviZ library, the other two being:

* [arviz-base](https://arviz-base.readthedocs.io/en/latest/) data related functionality, including converters from different PPLs.
* [arviz-stats](https://arviz-stats.readthedocs.io/en/latest/) for statistical functions and diagnostics.

We recommend most users install and use all three ArviZ components together through the main ArviZ package, as they're designed to work seamlessly as one toolkit. Advanced users may choose to install components individually for finer control over dependencies.

Note: All plotting functions - whether accessed through the full ArviZ package or directly via ArviZ-plots - are documented here.


## Exploratory Analysis of Bayesian Models

In Modern Bayesian statistics models are usually build and solve using probabilistic programming languages (PPLs) such as PyMC, Stan, NumPyro, etc. These languages allow users to specify models in a high-level language and perform inference using state-of-the-art algorithms like Markov Chain Monte Carlo (MCMC) or Variational Inference (VI). As a result we usually get a posterior distribution, in the form of samples. The posterior distribution has a central role in Bayesian statistics, but other distributions like the posterior and prior predictive distribution are also of interest. And other quantities may be relevant too.

The correct visualization, analysis, and interpretation of these computed data is key to properly answer the questions that motivate our analysis.

When working with Bayesian models there are a series of related tasks that need to be addressed besides inference itself:

* Diagnoses of the quality of the inference

* Model criticism, including evaluations of both model assumptions and model predictions

* Comparison of models, including model selection or model averaging

* Preparation of the results for a particular audience.

We call these tasks exploratory analysis of Bayesian models (EABM). Successfully performing such tasks are central to the iterative and interactive modelling process (See Bayesian Workflow). In the words of Persi Diaconis.

> Exploratory data analysis seeks to reveal structure, or simple descriptions in data. We look at numbers or graphs and try to find patterns. We pursue leads suggested by background information, imagination, patterns perceived, and experience with other data analyses.

The goal of ArviZ is to provide a unified interface for performing exploratory analysis of Bayesian models in Python, regardless of the PPL used to perform inference. This allows users to focus on the analysis and interpretation of the results, rather than on the details of the implementation.



(installation)=
## Installation

For instructions on how to install the full ArviZ package (including `arviz-base`, `arviz-stats` and `arviz-plots`), please refer to the [installation guide](https://python.arviz.org/en/latest/getting_started/Installation.html).

However, if you are only interested in the plotting functions provided by ArviZ-plots, please follow the instructions below:

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
xarray, numpy, arviz-base and arviz-stats.
None of the possible backends: matplotlib, bokeh or plotly are installed
by default.

Consequently, it is not recommended to install `arviz-plots` but
instead to choose which backend to use. For example `arviz-plots[bokeh]`
or `arviz-plots[matplotlib, plotly]`, multiple comma separated values are valid too.

This will ensure all relevant dependencies are installed. For example, to use the plotly backend,
both `plotly>5` and `webcolors` are required.

```{toctree}
:hidden:
:caption: User guide

tutorials/overview
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
:caption: Tutorials

ArviZ in Context <https://arviz-devs.github.io/EABM/>
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

BlueSky <https://bsky.app/profile/arviz.bsky.social>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/arviz-plots>
```
