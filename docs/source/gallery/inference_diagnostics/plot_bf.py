import matplotlib.pyplot as plt
from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-variat")

data = load_arviz_data("centered_eight")

pc = azp.plot_bf(
    data,
    backend="matplotlib",
    var_name="mu" 
)

plt.show()