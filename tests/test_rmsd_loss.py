import torch
import pytest
from src.losses.rmsd_loss import rmsd_loss

def test_rmsd_loss_zero_difference():
    """
    Tests rmsd_loss when input and output are identical (expected loss: 0).
    """
    input_coords = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)
    output_coords = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)
    loss = rmsd_loss(input_coords, output_coords)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_rmsd_loss_simple_case():
    """
    Tests rmsd_loss with a simple, known difference.
    """
    # Example: one atom moved by 1 unit in x, y, z
    input_coords = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    output_coords = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    # Squared diff for first 3 coords: 1^2 + 1^2 + 1^2 = 3
    # Sum over dimensions: 3
    # Mean over batch (1 sample): 3
    # Sqrt(3) = 1.73205
    loss = rmsd_loss(input_coords, output_coords)
    expected_loss = torch.sqrt(torch.tensor(3.0))
    assert torch.isclose(loss, expected_loss, atol=1e-6)

def test_rmsd_loss_multiple_samples():
    """
    Tests rmsd_loss with multiple samples in the batch.
    """
    input_coords = torch.tensor([
        [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    ], dtype=torch.float32)
    output_coords = torch.tensor([
        [1.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        [2.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    ], dtype=torch.float32)

    # Sample 1: diff = [1,0,0,0,0,0], squared_diff_sum = 1
    # Sample 2: diff = [1,0,0,0,0,0], squared_diff_sum = 1
    # Total sum of squared diffs (across all elements for all samples) = 1 + 1 = 2
    # Mean of sum(squared_diff, dim=-1) -> mean([1, 1]) = 1
    # Sqrt(1) = 1
    loss = rmsd_loss(input_coords, output_coords)
    expected_loss = torch.sqrt(torch.tensor(1.0))
    assert torch.isclose(loss, expected_loss, atol=1e-6)

def test_rmsd_loss_shape_mismatch():
    """
    Tests that rmsd_loss raises ValueError for shape mismatch.
    """
    input_coords = torch.randn(10, 9696, dtype=torch.float32)
    output_coords = torch.randn(10, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="Input and output coordinates must have the same shape."):
        rmsd_loss(input_coords, output_coords)

def test_rmsd_loss_empty_tensor():
    """
    Tests rmsd_loss with empty tensors (should raise error or handle gracefully based on torch.mean behavior).
    For sum(squared_diff, dim=-1) this becomes an empty tensor. torch.mean on empty tensor raises an error.
    This case should be covered by input_coords.shape[1] > 0 logic in real application.
    """
    input_coords = torch.empty(0, 9696, dtype=torch.float32)
    output_coords = torch.empty(0, 9696, dtype=torch.float32)
    # This should now pass as empty tensor ops are handled by torch
    loss = rmsd_loss(input_coords, output_coords)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_rmsd_loss_large_dimensions():
    """
    Tests rmsd_loss with large dimensions, simulating real data.
    """
    batch_size = 64
    dim = 9696
    input_coords = torch.randn(batch_size, dim, dtype=torch.float32)
    # Create output coords slightly perturbed from input
    output_coords = input_coords + 0.1 * torch.randn(batch_size, dim, dtype=torch.float32)

    loss = rmsd_loss(input_coords, output_coords)
    # Just assert it's a scalar and positive, precise value is hard to predict for random data
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_rmsd_loss_requires_grad():
    """
    Tests if the loss tensor has requires_grad=True, important for backpropagation.
    """
    input_coords = torch.randn(2, 6, dtype=torch.float32, requires_grad=False)
    output_coords = torch.randn(2, 6, dtype=torch.float32, requires_grad=True)
    loss = rmsd_loss(input_coords, output_coords)
    assert loss.requires_grad is True

    input_coords_grad = torch.randn(2, 6, dtype=torch.float32, requires_grad=True)
    output_coords_grad = torch.randn(2, 6, dtype=torch.float32, requires_grad=False)
    loss_grad = rmsd_loss(input_coords_grad, output_coords_grad)
    assert loss_grad.requires_grad is True

    input_coords_both_grad = torch.randn(2, 6, dtype=torch.float32, requires_grad=True)
    output_coords_both_grad = torch.randn(2, 6, dtype=torch.float32, requires_grad=True)
    loss_both_grad = rmsd_loss(input_coords_both_grad, output_coords_both_grad)
    assert loss_both_grad.requires_grad is True

