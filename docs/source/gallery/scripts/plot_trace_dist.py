from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")  # matplotlib only

data = load_arviz_data("non_centered_eight")
pc = azp.plot_trace_dist(
    data,
    backend="none"  # change to preferred backend
)
pc.show()
