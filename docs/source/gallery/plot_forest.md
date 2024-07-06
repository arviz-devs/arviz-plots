(gallery_forest)=
# Forest plot


:::{seealso}
API Documentation: {func}`~arviz_plots.plot_forest`
:::

```{jupyter-execute} scripts/plot_forest.py
:hide-output:
```

:::::{tab-set}
:class: full-width

::::{tab-item} Matplotlib

```{jupyter-execute}
:hide-code:

pc = azp.plot_forest(data, var_names=["home", "atts", "defs"], backend="matplotlib")
pc.show()
```
::::

::::{tab-item} Bokeh

```{bokeh-plot}
:source-position: none

from arviz_base import load_arviz_data
import arviz_plots as azp
from bokeh.plotting import show

data = load_arviz_data("rugby")
pc = azp.plot_forest(data, var_names=["home", "atts", "defs"], backend="bokeh")

show(pc.viz["chart"].item())
```
::::

::::{tab-item} Plotly

```{jupyter-execute}
:hide-code:

pc = azp.plot_forest(data, var_names=["home", "atts", "defs"], backend="plotly")
pc.show()
```
::::
:::::
