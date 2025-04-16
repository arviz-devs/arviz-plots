"""
# Bayes_factor

Compute Bayes factor using Savage–Dickey ratio. 

We can apply this function when the null model is nested within the alternative.
In other words when the null (``ref_val``) is a particular value of the model we are
building (see [here](https://statproofbook.github.io/P/bf-sddr.html)).

For others cases computing Bayes factor is not straightforward and requires more complex
methods. Instead, of Bayes factors, we usually recommend Pareto smoothed
importance sampling leave one out cross validation (PSIS-LOO-CV). In ArviZ, you will find
them as functions with ``loo`` in their names.

---
:::{seealso}
API Documentation: {func}`~arviz_plots.plot_bf`
:::
"""
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")

pc = azp.plot_bf(
    data,
    backend="none",  # change to preferred backend
    var_name="mu"
)

pc.show()