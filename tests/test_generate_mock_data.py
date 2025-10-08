import numpy as np
import pytest
from generate_mock_data import generate_mock_data

def test_generate_mock_data_shapes():
    """
    Test to verify that generate_mock_data returns arrays with the correct shapes.
    """
    print("\nRunning test_generate_mock_data_shapes...")
    input_coords, cv_coords = generate_mock_data()

    expected_input_shape = (10000, 9696)
    expected_cv_shape = (10000, 3)

    assert input_coords.shape == expected_input_shape, \
        f"Input coordinates shape mismatch. Expected {expected_input_shape}, got {input_coords.shape}"
    assert cv_coords.shape == expected_cv_shape, \
        f"CV coordinates shape mismatch. Expected {expected_cv_shape}, got {cv_coords.shape}"
    print(f"Shapes verified: input_coords {input_coords.shape}, cv_coords {cv_coords.shape}")

def test_generate_mock_data_types():
    """
    Test to verify that generate_mock_data returns arrays with the correct data types.
    """
    print("\nRunning test_generate_mock_data_types...")
    input_coords, cv_coords = generate_mock_data()

    assert input_coords.dtype == np.float32, \
        f"Input coordinates data type mismatch. Expected np.float32, got {input_coords.dtype}"
    assert cv_coords.dtype == np.float32, \
        f"CV coordinates data type mismatch. Expected np.float32, got {cv_coords.dtype}"
    print(f"Data types verified: input_coords {input_coords.dtype}, cv_coords {cv_coords.dtype}")

def test_generate_mock_data_ranges():
    """
    Test to verify that generated data falls within plausible, albeit mock, ranges.
    This ensures that the mock data has 'reasonable numerical values' as per the acceptance criteria.
    """
    print("\nRunning test_generate_mock_data_ranges...")
    input_coords, cv_coords = generate_mock_data()

    # Expected range for input_coords: -100 to 100
    assert np.all(input_coords >= -100) and np.all(input_coords < 100), \
        "Input coordinates values are outside the expected range [-100, 100)."
    # Expected range for cv_coords: 0 to 50
    assert np.all(cv_coords >= 0) and np.all(cv_coords < 50), \
        "CV coordinates values are outside the expected range [0, 50)."
    print("Data ranges verified successfully.")

if __name__ == "__main__":
    pytest.main([__file__])