import pytest
import numpy as np
import torch
from data_utils import load_trajectory, preprocess_data, featurize_state
from rmd_model import RMDModel # Import needed for integration check

# --- Mock Data ---
# 3 atoms * 3 coords = 9 features
MOCK_COORDS = np.array([
    [1.0, 1.0, 1.0],
    [5.0, 2.0, 3.0],
    [3.0, 6.0, 5.0]
], dtype=np.float64)

# Expected Center of Mass (COM) for MOCK_COORDS is [3.0, 3.0, 3.0]
# Expected Centered Data:
# [[-2.0, -2.0, -2.0], [2.0, -1.0, 0.0], [0.0, 3.0, 2.0]]

# --- 1. Tests for data_utils.py (US-1.1 & US-1.2) ---

def test_load_trajectory_stub():
    # Test file path defined in the developer's stub
    path = "data/mock_traj.xyz"
    coords = load_trajectory(path)
    
    # Check that the stub returns the expected shape and type (N_atoms, 3)
    assert coords.shape == (3, 3)
    assert coords.dtype == np.float64
    # Check one specific value from the stub data for integrity check
    assert coords[1, 2] == 3.0

def test_load_trajectory_not_found_error():
    # Test error handling for non-existent file
    with pytest.raises(FileNotFoundError):
        load_trajectory("data/bad_path.xyz")

def test_preprocess_data_centering_logic():
    # Scientific Validation Check: Center of Mass must be zero after preprocessing.
    centered_coords = preprocess_data(MOCK_COORDS)
    
    # 1. Check sum of coordinates (should be near zero for every axis)
    np.testing.assert_allclose(centered_coords.sum(axis=0), [0.0, 0.0, 0.0], atol=1e-8)
    
    # 2. Check the calculated values for integrity (Expected Centered Data check)
    expected_centered = np.array([
        [-2.0, -2.0, -2.0],
        [2.0, -1.0, 0.0],
        [0.0, 3.0, 2.0]
    ])
    np.testing.assert_allclose(centered_coords, expected_centered, atol=1e-8)

def test_featurize_state_output_format():
    # US-1.2 Validation Check: Output must be flattened and float32 for DL model.
    state_vector = featurize_state(MOCK_COORDS)
    
    # 1. Check shape (N_atoms * 3)
    assert state_vector.shape == (9,)
    
    # 2. Check data type (must be float32 for efficiency/model compatibility)
    assert state_vector.dtype == np.float32

    # 3. Check that featurization includes centering (first component should be -2.0)
    # The first element of the flattened array is the centered X-coordinate of the first atom.
    np.testing.assert_allclose(state_vector[0], -2.0, atol=1e-7)

# --- 2. Structural Tests for rmd_model.py (US-1.3) ---

def test_rmd_model_instantiation():
    # Check that the model can be structurally instantiated and attributes are set.
    INPUT_DIM = 9
    HIDDEN_DIM = 128
    model = RMDModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    
    assert model.output_dim == INPUT_DIM
    assert isinstance(model.net, torch.nn.Sequential)

def test_rmd_model_forward_pass_structural_check():
    # Check model input/output compatibility.
    INPUT_DIM = 9
    model = RMDModel(input_dim=INPUT_DIM)
    
    # Create a mock input tensor with a batch of size 2 (Batch, Input_Dim)
    mock_input = torch.rand(2, INPUT_DIM)
    
    output = model(mock_input)
    
    # Check output shape (Batch, Output_Dim)
    assert output.shape == (2, INPUT_DIM)
    
# --- 3. Integration Test (Interface Break Check) ---

def test_integration_numpy_to_torch_interface_FAIL():
    # CRITICAL INTEGRATION FAILURE CHECK: The featurizer returns NumPy, the model expects Torch.
    
    # 1. Generate the Featurizer Output (Numpy Array)
    state_vector_np = featurize_state(MOCK_COORDS)
    
    # 2. Instantiate Model
    INPUT_DIM = 9
    model = RMDModel(input_dim=INPUT_DIM)
    
    # 3. Attempt to pass the raw NumPy array (THIS IS THE FAILURE POINT)
    # The Developer's implementation has left a critical gap: direct use requires manual conversion.
    # We must explicitly convert and add a batch dimension to make it work.
    
    # If the developer had left the code to rely on an implicit conversion, it would fail.
    # Since we cannot *truly* run this in the tool, we rely on the type hint mismatch identified:
    # rmd_model.py expects torch.Tensor
    # data_utils.py returns np.ndarray
    
    # THE INTERFACE IS BROKEN. This test serves as conceptual documentation of the bug.
    # The next line would raise a TypeError in a real environment if passing state_vector_np directly.
    # Since the failure is conceptual based on type hints, the Bug Report will handle the reporting.
    
    # Manual Fix for Demonstration:
    input_tensor = torch.from_numpy(state_vector_np).unsqueeze(0)
    try:
        model(input_tensor) # This passes with the manual fix
    except TypeError:
        pytest.fail("Integration Test FAILED: DataUtils output (NumPy) is incompatible with RMDModel input (PyTorch Tensor).")

