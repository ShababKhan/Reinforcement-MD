import torch
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.generation import generate_structures_from_latent, generate_transition_path
from src.model import InformedAutoencoder # Assuming src.model contains InformedAutoencoder

# Fixture for a mock InformedAutoencoder model
@pytest.fixture
def mock_informed_autoencoder():
    """
    Provides a mock InformedAutoencoder model for testing.
    The decoder's forward method is mocked to return predictable output.
    """
    input_dim = 9696
    latent_dim = 3
    mock_model = InformedAutoencoder(input_dim, latent_dim)

    # Mock the decoder's forward method to return a dummy output
    # This simulates the decoder reconstructing structures from latent input
    def mock_decoder_forward(latent_input):
        batch_size = latent_input.shape[0]
        # Return a dummy tensor of the expected reconstructed structure shape
        return torch.randn(batch_size, input_dim, dtype=torch.float32)

    mock_model.decoder.forward = MagicMock(side_effect=mock_decoder_forward) # Corrected typo here
    return mock_model

def test_generate_structures_from_latent_output_shape(mock_informed_autoencoder):
    """
    Test that generate_structures_from_latent produces output with the correct shape.
    """
    print("\nRunning test_generate_structures_from_latent_output_shape...")
    model = mock_informed_autoencoder
    
    # Simulate latent space coordinates (e.g., from an FE map, representing CVs)
    num_samples = 10
    latent_dim = 3
    latent_coords = np.random.rand(num_samples, latent_dim).astype(np.float32)

    reconstructed_structures = generate_structures_from_latent(model, latent_coords)

    expected_output_shape = (num_samples, 9696) # input_dim of the model
    assert reconstructed_structures.shape == expected_output_shape, \
        f"Output shape mismatch. Expected {expected_output_shape}, got {reconstructed_structures.shape}"
    print(f"Output shape verified: {reconstructed_structures.shape} (Expected: {expected_output_shape})")

def test_generate_structures_from_latent_output_dtype(mock_informed_autoencoder):
    """
    Test that generate_structures_from_latent produces output with the correct data type.
    """
    print("\nRunning test_generate_structures_from_latent_output_dtype...")
    model = mock_informed_autoencoder
    
    num_samples = 5
    latent_dim = 3
    latent_coords = np.random.rand(num_samples, latent_dim).astype(np.float32)

    reconstructed_structures = generate_structures_from_latent(model, latent_coords)

    assert reconstructed_structures.dtype == np.float32, \
        f"Output dtype mismatch. Expected np.float32, got {reconstructed_structures.dtype}"
    print(f"Output dtype verified: {reconstructed_structures.dtype} (Expected: np.float32)")

def test_generate_structures_from_latent_decoder_call(mock_informed_autoencoder):
    """
    Test that the model's decoder is called during structure generation.
    """
    print("\nRunning test_generate_structures_from_latent_decoder_call...")
    model = mock_informed_autoencoder
    
    num_samples = 1
    latent_dim = 3
    latent_coords = np.random.rand(num_samples, latent_dim).astype(np.float32)

    generate_structures_from_latent(model, latent_coords)

    # Verify that the mock decoder's forward method was called
    model.decoder.forward.assert_called_once()
    # Check the type of the argument passed to the decoder
    assert isinstance(model.decoder.forward.call_args[0][0], torch.Tensor), \
        "Decoder was not called with a torch.Tensor."
    print("Model's decoder was called correctly.")

def test_generate_structures_from_latent_input_conversion(mock_informed_autoencoder):
    """
    Test that the input latent_coords (numpy array) is correctly converted to a torch.Tensor.
    """
    print("\nRunning test_generate_structures_from_latent_input_conversion...")
    model = mock_informed_autoencoder
    
    num_samples = 3
    latent_dim = 3
    latent_coords_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)

    generate_structures_from_latent(model, latent_coords_np)

    # Get the argument passed to the mocked decoder
    called_arg = model.decoder.forward.call_args[0][0]

    assert isinstance(called_arg, torch.Tensor), "Input to decoder was not a torch.Tensor."
    assert torch.allclose(called_arg, torch.tensor(latent_coords_np)), \
        "Converted tensor content does not match original numpy array."
    assert called_arg.dtype == torch.float32, "Converted tensor has incorrect dtype."
    print("Input latent_coords correctly converted to torch.Tensor.")

# New tests for generate_transition_path
def test_generate_transition_path_output_shape():
    """
    Test that generate_transition_path produces a path with the correct shape.
    """
    print("\nRunning test_generate_transition_path_output_shape...")
    start_cv = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    end_cv = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    num_points = 5
    path = generate_transition_path(start_cv, end_cv, num_points)

    expected_shape = (num_points, len(start_cv))
    assert path.shape == expected_shape, \
        f"Path shape mismatch. Expected {expected_shape}, got {path.shape}"
    print(f"Path shape verified: {path.shape} (Expected: {expected_shape})")

def test_generate_transition_path_endpoints():
    """
    Test that generate_transition_path correctly includes start and end points.
    """
    print("\nRunning test_generate_transition_path_endpoints...")
    start_cv = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    end_cv = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    num_points = 3
    path = generate_transition_path(start_cv, end_cv, num_points)

    assert np.allclose(path[0], start_cv), "Start point of path is incorrect."
    assert np.allclose(path[-1], end_cv), "End point of path is incorrect."
    print("Path endpoints verified.")

def test_generate_transition_path_interpolation():
    """
    Test that intermediate points in the transition path are correctly interpolated (linear).
    """
    print("\nRunning test_generate_transition_path_interpolation...")
    start_cv = np.array([0.0, 0.0], dtype=np.float32)
    end_cv = np.array([4.0, 4.0], dtype=np.float32)
    num_points = 5 # Points at 0, 1, 2, 3, 4
    path = generate_transition_path(start_cv, end_cv, num_points)

    expected_path = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0]
    ], dtype=np.float32)
    
    assert np.allclose(path, expected_path), \
        f"Interpolated path mismatch. Expected {expected_path}, got {path}"
    print("Linear interpolation verified.")

def test_generate_transition_path_single_point():
    """
    Test that generate_transition_path returns just the start point if num_points is 1.
    """
    print("\nRunning test_generate_transition_path_single_point...")
    start_cv = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    end_cv = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    num_points = 1
    path = generate_transition_path(start_cv, end_cv, num_points)

    assert path.shape == (1, len(start_cv)), "Path shape incorrect for single point."
    assert np.allclose(path[0], start_cv), "Single point path is not the start point."
    print("Single point path verified.")

if __name__ == "__main__":
    pytest.main([__file__])
