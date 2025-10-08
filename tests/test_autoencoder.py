import torch
import torch.nn as nn
import pytest
from src.models.autoencoder import BasicAutoencoder, Swish


class TestSwish:
    """
    Unit tests for the Swish activation function.
    """
    def test_swish_output_shape(self):
        """
        Tests if Swish output has the same shape as input.
        """
        swish = Swish()
        input_tensor = torch.randn(10, 5, dtype=torch.float32)
        output_tensor = swish(input_tensor)
        assert output_tensor.shape == input_tensor.shape

    def test_swish_values(self):
        """
        Tests Swish output values with known inputs.
        Swish(x) = x * sigmoid(x)
        """
        swish = Swish()
        # Test with 0
        input_zero = torch.tensor([0.0], dtype=torch.float32)
        output_zero = swish(input_zero)
        assert torch.isclose(output_zero, torch.tensor(0.0), atol=1e-6)

        # Test with positive value (e.g., 1.0)
        input_one = torch.tensor([1.0], dtype=torch.float32)
        expected_one = 1.0 * torch.sigmoid(torch.tensor(1.0))
        output_one = swish(input_one)
        assert torch.isclose(output_one, expected_one, atol=1e-6)

        # Test with negative value (e.g., -1.0)
        input_neg_one = torch.tensor([-1.0], dtype=torch.float32)
        expected_neg_one = -1.0 * torch.sigmoid(torch.tensor(-1.0))
        output_neg_one = swish(input_neg_one)
        assert torch.isclose(output_neg_one, expected_neg_one, atol=1e-6)

    def test_swish_gradient(self):
        """
        Tests if Swish function supports gradient computation.
        """
        swish = Swish()
        input_tensor = torch.randn(5, 5, requires_grad=True, dtype=torch.float32)
        output_tensor = swish(input_tensor)
        output_tensor.sum().backward()
        assert input_tensor.grad is not None


class TestBasicAutoencoder:
    """
    Unit tests for the BasicAutoencoder class.
    """
    def test_autoencoder_architecture(self):
        """
        Tests if the autoencoder has the correct layer sizes and structure.
        """
        input_dim = 9696
        latent_dim = 3
        model = BasicAutoencoder(input_dim, latent_dim)

        # Check encoder layers
        assert len(model.encoder) == 8  # 4 Linear + 4 Swish
        assert isinstance(model.encoder[0], nn.Linear)
        assert model.encoder[0].in_features == input_dim
        assert model.encoder[0].out_features == 5000
        assert isinstance(model.encoder[1], Swish)
        assert isinstance(model.encoder[6], nn.Linear)
        assert model.encoder[6].in_features == 500
        assert model.encoder[6].out_features == latent_dim

        # Check decoder layers
        assert len(model.decoder) == 8  # 4 Linear + 4 Swish
        assert isinstance(model.decoder[0], nn.Linear)
        assert model.decoder[0].in_features == latent_dim
        assert model.decoder[0].out_features == 500
        assert isinstance(model.decoder[1], Swish)
        assert isinstance(model.decoder[6], nn.Linear)
        assert model.decoder[6].in_features == 5000
        assert model.decoder[6].out_features == input_dim

    def test_autoencoder_forward_pass_shapes(self):
        """
        Tests if the forward pass produces outputs with the correct shapes.
        """
        input_dim = 9696
        latent_dim = 3
        batch_size = 64
        model = BasicAutoencoder(input_dim, latent_dim)
        mock_input = torch.randn(batch_size, input_dim, dtype=torch.float32)

        latent_output, reconstructed_output = model(mock_input)

        assert latent_output.shape == (batch_size, latent_dim)
        assert reconstructed_output.shape == (batch_size, input_dim)

    def test_autoencoder_forward_pass_values(self):
        """
        Tests if the forward pass produces finite output values.
        """
        input_dim = 9696
        latent_dim = 3
        batch_size = 64
        model = BasicAutoencoder(input_dim, latent_dim)
        mock_input = torch.randn(batch_size, input_dim, dtype=torch.float32)

        latent_output, reconstructed_output = model(mock_input)

        assert torch.isfinite(latent_output).all()
        assert torch.isfinite(reconstructed_output).all()

    def test_autoencoder_default_dimensions(self):
        """
        Tests autoencoder with default input_dim and latent_dim.
        """
        model = BasicAutoencoder()
        assert model.input_dim == 9696
        assert model.latent_dim == 3

        batch_size = 1
        mock_input = torch.randn(batch_size, 9696, dtype=torch.float32)
        latent_output, reconstructed_output = model(mock_input)
        assert latent_output.shape == (batch_size, 3)
        assert reconstructed_output.shape == (batch_size, 9696)

    def test_autoencoder_gradient(self):
        """
        Tests if the autoencoder model supports gradient computation.
        """
        input_dim = 9696
        latent_dim = 3
        batch_size = 2
        model = BasicAutoencoder(input_dim, latent_dim)
        mock_input = torch.randn(batch_size, input_dim, requires_grad=True, dtype=torch.float32)

        latent_output, reconstructed_output = model(mock_input)
        # A simple loss to enable backward pass
        loss = reconstructed_output.sum()
        loss.backward()

        assert mock_input.grad is not None
        for param in model.parameters():
            assert param.grad is not None
