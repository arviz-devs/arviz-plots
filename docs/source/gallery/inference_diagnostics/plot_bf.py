"""
# Bayes_factor

Compute Bayes factor using Savage–Dickey ratio

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_bf`
:::
"""
import arviz_plots as azp
from arviz_base import load_arviz_data

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")

pc = azp.plot_bf(
    data,
    backend="none",
    var_name="mu"
)

pc.show()
