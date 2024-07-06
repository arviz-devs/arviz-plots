(gallery_trace_dist)=
# Trace and distribution plot


:::{seealso}
API Documentation: {func}`~arviz_plots.plot_trace_dist`
:::

```{jupyter-execute} scripts/plot_trace_dist.py
:hide-output:
```

:::::{tab-set}
:class: full-width

::::{tab-item} Matplotlib

```{jupyter-execute}
:hide-code:

pc = azp.plot_trace_dist(data, backend="matplotlib")
pc.show()
```
::::

::::{tab-item} Bokeh

```{bokeh-plot}
:source-position: none

from arviz_base import load_arviz_data
import arviz_plots as azp
from bokeh.plotting import show


data = load_arviz_data("non_centered_eight")

pc = azp.plot_trace_dist(data, backend="bokeh")
show(pc.viz["chart"].item())
```
::::

::::{tab-item} Plotly

```{jupyter-execute}
:hide-code:

pc = azp.plot_trace_dist(data, backend="plotly")
pc.show()
```
::::
:::::
