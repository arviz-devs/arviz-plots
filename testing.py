import arviz as az
import arviz_plots as azp
from bokeh.io import show
import matplotlib.pyplot as plt

# 1. Load Data
try:
    print("Loading glycan_torsion_angles data...")
    data = az.load_arviz_data("glycan_torsion_angles")
except Exception as e:
    print(f"Data Load Error: {e}. Try running 'pip install netCDF4'")
    exit()

# 2. Run Plot
try:
    print("Initializing plot_trace with circ_var_names...")
    # This calls the code you just modified
    pc = azp.plot_trace(
        data, 
        var_names=["tors"], 
        circ_var_names=["tors"], 
        backend="bokeh"
    )

    print("Success! Opening Bokeh tab...")
    show(pc.backend_config["figure"])

except ValueError as v:
    print(f"STILL GETTING VALUE ERROR: {v}")
    print("Check if you saved the changes in BOTH trace_plot.py and trace_dist_plot.py")
except Exception as e:
    print(f"An unexpected error occurred: {e}")