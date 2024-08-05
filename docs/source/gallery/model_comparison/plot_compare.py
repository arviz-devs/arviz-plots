"""
(gallery_forest_pp_obs)=
# Posterior predictive and observations forest plot

Overlay of forest plot for the posterior predictive samples and the actual observations

---

:::{seealso}
API Documentation: {func}`~arviz_plots.plot_forest`

Other gallery examples using `plot_forest`: {ref}`gallery_forest`, {ref}`gallery_forest_shade`
:::
"""
from importlib import import_module

from arviz_base import load_arviz_data

import arviz_plots as azp

azp.style.use("arviz-clean")

backend="none"  # change to preferred backend

cmp_df = pd.DataFrame({"elpd_loo": [-4.575778, -14.309050, -16], 
                       "p_loo": [2.646204, 2.399241, 2], 
                       "elpd_diff": [0.000000, 9.733272, 11], 
                       "weight": [1.000000e+00, 3.215206e-13, 0], 
                       "se": [2.318739, 2.673219, 2], 
                       "dse": [0.00000, 2.68794, 2], 
                       "warning": [False, False, False], 
                       "scale": ["log", "log", "log"]}, index=["modelo_p", "modelo_l", "modelo_d"])

azp.plot_compare(cmp_df, backend=backend)