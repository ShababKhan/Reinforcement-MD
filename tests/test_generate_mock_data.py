import numpy as np
import pytest
from generate_mock_data import generate_mock_data

def test_generate_mock_data_shapes():
    """
    Tests if generate_mock_data produces arrays with the correct shapes.
    """
    input_coords, cv_coords = generate_mock_data()
    assert input_coords.shape == (10000, 9696)
    assert cv_coords.shape == (10000, 3)

def test_generate_mock_data_datatypes():
    """
    Tests if generate_mock_data produces arrays with the correct data types.
    """
    input_coords, cv_coords = generate_mock_data()
    assert input_coords.dtype == np.float32
    assert cv_coords.dtype == np.float32

def test_generate_mock_data_value_range():
    """
    Tests if generate_mock_data produces arrays with values between 0 and 1 (inclusive).
    """
    input_coords, cv_coords = generate_mock_data()
    assert np.all(input_coords >= 0) and np.all(input_coords <= 1)
    assert np.all(cv_coords >= 0) and np.all(cv_coords <= 1)

def test_generate_mock_data_not_empty():
    """
    Tests if the generated arrays are not empty.
    """
    input_coords, cv_coords = generate_mock_data()
    assert input_coords.size > 0
    assert cv_coords.size > 0
