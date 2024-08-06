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
import arviz_plots as azp

azp.style.use("arviz-clean")

backend="none"  # change to preferred backend

cmp_df = pd.DataFrame({"elpd_loo": [-4.5, -14.3, -16.2], 
                       "p_loo": [2.6, 2.3, 2.1], 
                       "elpd_diff": [0, 9.7, 11.3], 
                       "weight": [0.9, 0.1, 0], 
                       "se": [2.3, 2.7, 2.3], 
                       "dse": [0, 2.7, 2.3], 
                       "warning": [False, False, False], 
                       "scale": ["log", "log", "log"]}, index=["Model B", "Model A", "Model C"])

azp.plot_compare(cmp_df, backend=backend)