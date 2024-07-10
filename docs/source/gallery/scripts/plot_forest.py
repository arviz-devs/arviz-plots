from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")  # matplotlib only

data = load_arviz_data("rugby")
pc = azp.plot_forest(
    data,
    var_names=["home", "atts", "defs"],
    backend="none"  # change to preferred backend
)
pc.show()
