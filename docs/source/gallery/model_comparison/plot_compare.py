"""
# Predicive model comparison plot

Compare multiple models using predictive accuracy estimates like  {abbr}`LOO-CV (leave one out cross validation)` or {abbr}`WAIC (widely applicable information criterion)`

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_compare`
:::
"""
import arviz_plots as azp
import pandas as pd

azp.style.use("arviz-clean")


cmp_df = pd.DataFrame({"elpd_loo": [-4.5, -14.3, -16.2], 
                       "p_loo": [2.6, 2.3, 2.1], 
                       "elpd_diff": [0, 9.7, 11.3], 
                       "weight": [0.9, 0.1, 0], 
                       "se": [2.3, 2.7, 2.3], 
                       "dse": [0, 2.7, 2.3], 
                       "warning": [False, False, False], 
                       "scale": ["log", "log", "log"]}, index=["Model B", "Model A", "Model C"])

pc = azp.plot_compare(cmp_df, backend="none")  # change to preferred backend
pc.show()