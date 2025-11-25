
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from contribution.joint_plot import plot_joint

def test_joint_plot():
    # Generate synthetic data
    # 2 chains, 100 draws, 2 variables (mu, tau)
    rng = np.random.default_rng(42)
    mu = rng.normal(0, 1, size=(2, 100))
    tau = rng.exponential(1, size=(2, 100))
    
    data = xr.DataTree.from_dict({
        "posterior": xr.Dataset(
            {
                "mu": (("chain", "draw"), mu),
                "tau": (("chain", "draw"), tau),
            },
            coords={
                "chain": [0, 1],
                "draw": np.arange(100),
            }
        )
    })
    
    print("Plotting joint plot with synthetic data...")
    pc = plot_joint(
        data,
        var_names=["mu", "tau"],
        backend="matplotlib"
    )
    
    # Save figure
    print("Saving figure...")
    pc.viz["figure"].item().savefig("joint_plot_test.png")
    print("Done.")

if __name__ == "__main__":
    test_joint_plot()
