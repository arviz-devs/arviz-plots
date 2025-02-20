import numpy as np
import arviz as az
import pytest
import matplotlib.pyplot as plt  

from arviz_plots.plots.bfplot import plot_bf

@pytest.fixture
def create_test_idata():
    """Fixture to create a test InferenceData object."""
    return az.from_dict(
        posterior={"a": np.random.normal(1, 0.5, 5000)},
        prior={"a": np.random.normal(0, 1, 5000)}
    )

def test_plot_bf_basic(create_test_idata):
    """Test basic functionality of plot_bf with default parameters."""
    idata = create_test_idata
    result, axes = plot_bf(idata, var_name="a", ref_val=0, show=False)
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0
    assert axes is not None
    plt.close('all')  

def test_plot_bf_with_custom_ref_val(create_test_idata):
    """Test plot_bf with a custom reference value."""
    idata = create_test_idata
    result, axes = plot_bf(idata, var_name="a", ref_val=1.0, show=False)
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0
    assert axes is not None
    plt.close('all')  

def test_plot_bf_with_custom_colors(create_test_idata):
    """Test plot_bf with custom colors."""
    idata = create_test_idata
    result, axes = plot_bf(idata, var_name="a", ref_val=0, colors=("red", "blue"), show=False)
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0
    assert axes is not None
    plt.close('all')  

def test_plot_bf_with_custom_prior(create_test_idata):
    """Test plot_bf with a custom prior."""
    idata = create_test_idata
    custom_prior = np.random.normal(0.5, 0.2, 5000)
    result, axes = plot_bf(idata, var_name="a", prior=custom_prior, ref_val=0, show=False)
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0
    assert axes is not None
    plt.close('all') 

def test_plot_bf_with_custom_figsize(create_test_idata):
    """Test plot_bf with a custom figure size."""
    idata = create_test_idata
    result, axes = plot_bf(idata, var_name="a", ref_val=0, figsize=(10, 6), show=False)
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0
    assert axes is not None
    plt.close('all')  

def test_plot_bf_with_custom_backend_kwargs(create_test_idata):
    """Test plot_bf with custom backend kwargs."""
    idata = create_test_idata
    result, axes = plot_bf(idata, var_name="a", ref_val=0, backend_kwargs={"dpi": 100}, show=False)
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0
    assert axes is not None
    plt.close('all')  
    
def test_plot_bf_show(create_test_idata):
    """Test plot_bf with the show parameter."""
    idata = create_test_idata
    result, axes = plot_bf(idata, var_name="a", ref_val=0, show=False)  
    assert "BF10" in result
    assert "BF01" in result
    assert result["BF10"] > 0
    assert result["BF01"] > 0
    assert axes is not None
    plt.close('all')  
