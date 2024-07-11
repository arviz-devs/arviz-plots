from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")  # matplotlib only

c = load_arviz_data("centered_eight")
n = load_arviz_data("non_centered_eight")
pc = azp.plot_dist(
    {"Centered": c, "Non Centered": n},
    backend="none"  # change to preferred backend
)
pc.show()
