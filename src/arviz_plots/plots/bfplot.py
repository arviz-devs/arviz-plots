import logging
from arviz.plots.plot_utils import get_plotting_function

import arviz_base as azb
from arviz_stats import bayes_factor

_log = logging.getLogger(__name__)



def plot_bf(
    idata,
    var_name,
    prior=None,
    ref_val=0,
    colors=("C0", "C1"),
    figsize=None,
    textsize=None,
    hist_kwargs=None,
    plot_kwargs=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    r"""Approximated Bayes Factor for comparing hypothesis of two nested models.

    The Bayes factor is estimated by comparing a model (H1) against a model in which the
    parameter of interest has been restricted to be a point-null (H0). This computation
    assumes the models are nested and thus H0 is a special case of H1.

    Notes
    -----
    The bayes Factor is approximated as the Savage-Dickey density ratio
    algorithm presented in [1]_.

    Parameters
    ----------
    idata : InferenceData
        Any object that can be converted to an :class:`arviz.InferenceData` object.
    var_name : str
        Name of variable we want to test.
    prior : numpy.array, optional
        In case we want to use a different prior, for example for sensitivity analysis.
    ref_val : int, default 0
        Point-null for Bayes factor estimation.
    colors : tuple, default ('C0', 'C1')
        Tuple of valid Matplotlib colors. First element for the prior, second for the posterior.
    figsize : (float, float), optional
        Figure size. If `None`, it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines.
    plot_kwargs : dict, optional
        Additional keywords passed to :func:`matplotlib.pyplot.plot`.
    hist_kwargs : dict, optional
        Additional keywords passed to :func:`arviz.plot_dist`. Only works for discrete variables.
    ax : axes, optional
        :class:`matplotlib.axes.Axes` or :class:`bokeh.plotting.Figure`.
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    dict : A dictionary with BF10 (Bayes Factor 10 (H1/H0 ratio), and BF01 (H0/H1 ratio).
    axes : matplotlib_axes or bokeh_figure

    References
    ----------
    .. [1] Heck, D., 2019. A caveat on the Savage-Dickey density ratio:
       The case of computing Bayes factors for regression parameters.
    """


    if prior is None:
        prior = azb.extract(idata, var_names=var_name, group="prior").values


    try:
        bf, p_at_ref_val = bayes_factor.bayes_factor(
            idata, var_name, prior=prior, ref_val=ref_val, return_ref_vals=True
        )
    except AttributeError as e:
        _log.error("Error: %s", e)
        raise AttributeError("Bayes factor function not found in arviz.stats.")

    bf_10 = bf["BF10"]
    bf_01 = bf["BF01"]

    posterior = azb.extract(idata, var_names=var_name)


    bfplot_kwargs = dict(
        ax=ax,
        bf_10=bf_10.item(),
        bf_01=bf_01.item(),
        prior=prior,
        posterior=posterior,
        ref_val=ref_val,
        prior_at_ref_val=p_at_ref_val["prior"],
        posterior_at_ref_val=p_at_ref_val["posterior"],
        var_name=var_name,
        colors=colors,
        figsize=figsize,
        textsize=textsize,
        hist_kwargs=hist_kwargs,
        plot_kwargs=plot_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )


    plot = get_plotting_function("plot_bf", "bfplot", backend)
    axes = plot(**bfplot_kwargs)

    return {"BF10": bf_10, "BF01": bf_01}, axes
