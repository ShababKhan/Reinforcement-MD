import torch
import pytest
from unittest.mock import MagicMock, patch

# Assuming these imports are from the project structure
from src.model import BasicAutoencoder
from losses import rmsd_loss
from generate_mock_data import generate_mock_data
from src.trainer import train_basic_autoencoder

# Mocking the generate_mock_data function for controlled testing
@pytest.fixture
def mock_data():
    """Fixture to provide mock input and CV coordinates."""
    num_samples = 100
    input_dim = 10
    cv_dim = 2
    mock_input = torch.randn(num_samples, input_dim, dtype=torch.float32)
    mock_cv = torch.randn(num_samples, cv_dim, dtype=torch.float32) # Not directly used by basic autoencoder, but part of generate_mock_data output
    return mock_input, mock_cv

@pytest.fixture
def basic_autoencoder_model():
    """Fixture to provide a BasicAutoencoder instance."""
    input_dim = 10 # Must match mock_data
    latent_dim = 3
    return BasicAutoencoder(input_dim, latent_dim)

def test_train_basic_autoencoder_loss_decrease(mock_data, basic_autoencoder_model):
    """
    Test that the training loss decreases over a small number of epochs.
    """
    print("\nRunning test_train_basic_autoencoder_loss_decrease...")
    input_coords, _ = mock_data
    model = basic_autoencoder_model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss = float('inf')
    final_loss = float('inf')
    num_test_epochs = 5 # Small number of epochs for quick test

    # Capture losses over epochs
    losses = []
    for epoch in range(num_test_epochs):
        current_loss = train_basic_autoencoder(model, input_coords, optimizer, rmsd_loss, batch_size=10, epochs=1)
        losses.append(current_loss)
        print(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")

    # Check if loss generally decreases (allowing for some fluctuation in small epochs)
    # This is a simplified check for general trend, a more robust test might use a statistical method
    assert losses[0] >= losses[-1], "Training loss did not decrease or stayed constant over epochs."
    assert losses[-1] < losses[0], "Training loss did not decrease."
    print("Training loss showed a general decreasing trend over epochs.")


def test_train_basic_autoencoder_runs_without_error(mock_data, basic_autoencoder_model):
    """
    Test that the train_basic_autoencoder function runs without raising exceptions.
    """
    print("\nRunning test_train_basic_autoencoder_runs_without_error...")
    input_coords, _ = mock_data
    model = basic_autoencoder_model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    try:
        train_basic_autoencoder(model, input_coords, optimizer, rmsd_loss, batch_size=10, epochs=2)
        print("train_basic_autoencoder ran successfully without errors.")
    except Exception as e:
        pytest.fail(f"train_basic_autoencoder raised an unexpected exception: {e}")

def test_train_basic_autoencoder_optimizer_step_called(mock_data, basic_autoencoder_model):
    """
    Verify that the optimizer's step method is called during training.
    """
    print("\nRunning test_train_basic_autoencoder_optimizer_step_called...")
    input_coords, _ = mock_data
    model = basic_autoencoder_model
    optimizer = MagicMock(spec=torch.optim.Adam) # Mock the optimizer
    optimizer.param_groups = [{'params': list(model.parameters())}] # Needed for some internal optimizer logic

    train_basic_autoencoder(model, input_coords, optimizer, rmsd_loss, batch_size=10, epochs=1)

    optimizer.step.assert_called_once()
    print("Optimizer's step method was called.")

def test_train_basic_autoencoder_correct_number_of_batches(mock_data, basic_autoencoder_model):
    """
    Test that the correct number of batches are processed per epoch.
    """
    print("\nRunning test_train_basic_autoencoder_correct_number_of_batches...")
    input_coords, _ = mock_data # num_samples=100 from fixture
    model = basic_autoencoder_model
    optimizer = MagicMock(spec=torch.optim.Adam)
    optimizer.param_groups = [{'params': list(model.parameters())}]

    batch_size = 10
    num_samples = input_coords.shape[0] # 100
    expected_batches_per_epoch = (num_samples + batch_size - 1) // batch_size # Ceil division: 100/10 = 10

    # Mock model's forward pass and loss function
    with patch('src.model.BasicAutoencoder.forward', return_value=(torch.randn(batch_size, 3), torch.randn(batch_size, 10))):
        with patch('losses.rmsd_loss', return_value=torch.tensor(0.1)):
            train_basic_autoencoder(model, input_coords, optimizer, rmsd_loss, batch_size=batch_size, epochs=1)
            assert optimizer.zero_grad.call_count == expected_batches_per_per_epoch
            assert optimizer.step.call_count == expected_batches_per_epoch
            print(f"Correct number of batches ({expected_batches_per_epoch}) processed per epoch.")

if __name__ == "__main__":
    pytest.main([__file__])
