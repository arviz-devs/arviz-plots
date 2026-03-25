import arviz as az
import arviz_plots as azp
from bokeh.io import show
import matplotlib.pyplot as plt

try:
    # 1. Load the circular data
    print("Loading data...")
    data = az.load_arviz_data("glycan_torsion_angles")

    # 2. Call the trace plot with your new parameter
    print("Generating plot...")
    pc = azp.plot_trace(
        data, 
        var_names=["tors"], 
        circ_var_names=["tors"], 
        backend="bokeh"
    )

    # 3. Open the browser tab
    print("Opening Bokeh browser tab...")
    show(pc.backend_config["figure"])

except Exception as e:
    print(f"Error caught: {e}")