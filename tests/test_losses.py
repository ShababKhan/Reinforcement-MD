import torch
import pytest
from losses import rmsd_loss

def test_rmsd_loss_identical_inputs():
    """
    Test that rmsd_loss returns 0 when inputs are identical.
    """
    print("
Running test_rmsd_loss_identical_inputs...")
    y_pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_true = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    loss = rmsd_loss(y_pred, y_true)
    assert torch.isclose(loss, torch.tensor(0.0)), f"Expected loss 0.0, got {loss.item()}"
    print(f"Loss for identical inputs: {loss.item()} (Expected: 0.0)")

def test_rmsd_loss_different_inputs():
    """
    Test that rmsd_loss calculates a non-zero loss for different inputs.
    """
    print("
Running test_rmsd_loss_different_inputs...")
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
    print("
Running test_rmsd_loss_shape_mismatch...")
    y_pred = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    y_true = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="Input tensors y_pred and y_true must have the same shape."):
        rmsd_loss(y_pred, y_true)
    print("ValueError correctly raised for shape mismatch.")

def test_rmsd_loss_batch_size():
    """
    Test rmsd_loss with a batch of inputs.
    """
    print("
Running test_rmsd_loss_batch_size...")
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

if __name__ == "__main__":
    pytest.main([__file__])
