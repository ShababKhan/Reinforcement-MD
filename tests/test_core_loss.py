import numpy as np
import torch
import pytest
from src.core_loss_function import compute_custom_loss # Assuming successful import

def test_rmsd_calculation_correctness():
    # Manual check: RMSD should be 1.0 for this simple case
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    
    # Expected RMSD = sqrt(mean((1^2 + 1^2 + 1^2)/3)) = sqrt(1) = 1.0
    expected_loss = 1.0
    
    loss = compute_custom_loss(y_true, y_pred)
    assert isinstance(loss, float)
    assert np.isclose(loss, expected_loss)

def test_rmsd_zero_loss():
    # Test case where prediction equals truth
    y_true = np.array([5.0, 10.0])
    y_pred = np.array([5.0, 10.0])
    
    loss = compute_custom_loss(y_true, y_pred)
    assert np.isclose(loss, 0.0)

def test_output_type_is_float():
    # Ensure the return type matches the signature's promise
    y_true = np.array([1.0])
    y_pred = np.array([2.0])
    
    loss = compute_custom_loss(y_true, y_pred)
    assert isinstance(loss, float)

def test_error_handling_on_invalid_input():
    # Test case that forces the try/except block due to invalid data type
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([3.0, 'a']) # 'a' should cause failure during numpy/torch conversion
    
    with pytest.raises(TypeError, match="Error during loss computation"):
        compute_custom_loss(y_true, y_pred)