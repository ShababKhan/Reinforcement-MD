import torch
import pytest
from losses import rmsd_loss, latent_loss # Import both loss functions

def test_rmsd_loss_identical_inputs():
    """
    Test that rmsd_loss returns 0 when inputs are identical.
    """
    print("\nRunning test_rmsd_loss_identical_inputs...")
    y_pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_true = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    loss = rmsd_loss(y_pred, y_true)
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected loss 0.0, got {loss.item()}"
    print(f"Loss for identical inputs: {loss.item()} (Expected: 0.0)")

def test_rmsd_loss_different_inputs():
    """
    Test that rmsd_loss calculates a non-zero loss for different inputs.
    """
    print("\nRunning test_rmsd_loss_different_inputs...")
    y_pred = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    y_true = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32)
    # Expected: sqrt(((1-2)^2 + (2-3)^2 + (3-4)^2) / 3) = sqrt((1+1+1)/3) = sqrt(1) = 1.0
    loss = rmsd_loss(y_pred, y_true)
    assert torch.isclose(loss, torch.tensor(1.0)), f"Expected loss 1.0, got {loss.item()}"
    print(f"Loss for different inputs: {loss.item()} (Expected: 1.0)")

def test_rmsd_loss_shape_mismatch():
    """
    Test that rmsd_loss raises ValueError for shape mismatch.
    """
    print("\nRunning test_rmsd_loss_shape_mismatch...")
    y_pred = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    y_true = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="Input tensors y_pred and y_true must have the same shape."):
        rmsd_loss(y_pred, y_true)
    print("ValueError correctly raised for shape mismatch for rmsd_loss.")

def test_rmsd_loss_batch_size():
    """
    Test rmsd_loss with a batch of inputs.
    """
    print("\nRunning test_rmsd_loss_batch_size...")
    y_pred = torch.tensor([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], dtype=torch.float32)
    y_true = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=torch.float32)
    # Differences: [-1, -1, -1], [2, 2, 2]
    # Squared diffs: [1, 1, 1], [4, 4, 4]
    # Sum of squared diffs: 1+1+1+4+4+4 = 15
    # Total elements: 6
    # Mean squared diff: 15 / 6 = 2.5
    # RMSD-like loss: sqrt(2.5) approx 1.5811
    loss = rmsd_loss(y_pred, y_true)
    expected_loss = torch.sqrt(torch.tensor(2.5))
    assert torch.isclose(loss, expected_loss), f"Expected loss {expected_loss.item()}, got {loss.item()}"
    print(f"Loss for batch inputs: {loss.item()} (Expected: {expected_loss.item()})")

# New tests for latent_loss
def test_latent_loss_identical_inputs():
    """
    Test that latent_loss returns 0 when latent and CV coordinates are identical.
    """
    print("\nRunning test_latent_loss_identical_inputs...")
    ls_coords = torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.5, 2.0]], dtype=torch.float32)
    cv_coords = torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.5, 2.0]], dtype=torch.float32)
    loss = latent_loss(ls_coords, cv_coords)
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected latent loss 0.0, got {loss.item()}"
    print(f"Latent loss for identical inputs: {loss.item()} (Expected: 0.0)")

def test_latent_loss_different_inputs():
    """
    Test that latent_loss calculates a non-zero loss for different inputs.
    """
    print("\nRunning test_latent_loss_different_inputs...")
    ls_coords = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    cv_coords = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32)
    # Expected: sqrt(((1-2)^2 + (2-3)^2 + (3-4)^2) / 3) = sqrt((1+1+1)/3) = sqrt(1) = 1.0
    loss = latent_loss(ls_coords, cv_coords)
    assert torch.isclose(loss, torch.tensor(1.0)), f"Expected latent loss 1.0, got {loss.item()}"
    print(f"Latent loss for different inputs: {loss.item()} (Expected: 1.0)")

def test_latent_loss_shape_mismatch():
    """
    Test that latent_loss raises ValueError for shape mismatch.
    """
    print("\nRunning test_latent_loss_shape_mismatch...")
    ls_coords = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    cv_coords = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    with pytest.raises(ValueError, match="Input tensors ls_coords and cv_coords must have the same shape."):
        latent_loss(ls_coords, cv_coords)
    print("ValueError correctly raised for shape mismatch for latent_loss.")

if __name__ == "__main__":
    pytest.main([__file__])
