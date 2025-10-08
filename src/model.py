
import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x).
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class BasicAutoencoder(nn.Module):
    """
    A basic autoencoder model for protein structure prediction, as described in the paper.

    This autoencoder takes flattened Cartesian coordinates of protein structures as input,
    compresses them into a low-dimensional latent space through an encoder, and then
    reconstructs them back to the original Cartesian representation using a decoder.
    The architecture uses fully connected layers with Swish activation functions.

    The network layout is as follows (input_dim=9696, latent_dim=3):
    Encoder:
    - Linear(input_dim, 5000)
    - Swish activation
    - Linear(5000, 1000)
    - Swish activation
    - Linear(1000, 500)
    - Swish activation
    - Linear(500, latent_dim) (Latent Space)

    Decoder:
    - Linear(latent_dim, 500)
    - Swish activation
    - Linear(500, 1000)
    - Swish activation
    - Linear(1000, 5000)
    - Swish activation
    - Linear(5000, input_dim) (Output/Prediction Layer)

    @param input_dim: The dimensionality of the input protein structure (flattened Cartesian coordinates).
                      Defaults to 9696 as per the paper (CRBN heavy atoms).
    @param latent_dim: The dimensionality of the latent space. Defaults to 3 as per the paper (CV space).
    """
    def __init__(self, input_dim=9696, latent_dim=3):
        super(BasicAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 5000),
            Swish(),
            nn.Linear(5000, 1000),
            Swish(),
            nn.Linear(1000, 500),
            Swish(),
            nn.Linear(500, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500),
            Swish(),
            nn.Linear(500, 1000),
            Swish(),
            nn.Linear(1000, 5000),
            Swish(),
            nn.Linear(5000, input_dim)
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        @param x: Input tensor representing flattened protein coordinates.
                  Shape: (batch_size, input_dim)
        @return: A tuple containing:
                 - latent (torch.Tensor): The output from the latent space.
                                          Shape: (batch_size, latent_dim)
                 - reconstructed_x (torch.Tensor): The reconstructed protein coordinates.
                                                   Shape: (batch_size, input_dim)
        """
        latent = self.encoder(x)
        reconstructed_x = self.decoder(latent)
        return latent, reconstructed_x

if __name__ == "__main__":
    # Example usage and shape verification
    input_dim = 9696
    latent_dim = 3
    batch_size = 64

    model = BasicAutoencoder(input_dim, latent_dim)
    print(f"Basic Autoencoder Model: {model}")

    dummy_input = torch.randn(batch_size, input_dim) # Random input data
    latent_output, reconstructed_output = model(dummy_input)

    print(f"Shape of dummy input: {dummy_input.shape}")
    print(f"Shape of latent output: {latent_output.shape}")
    print(f"Shape of reconstructed output: {reconstructed_output.shape}")

    # Assertions for acceptance criteria (forward pass shapes)
    assert latent_output.shape == (batch_size, latent_dim), \
        f"Latent output shape mismatch. Expected {(batch_size, latent_dim)}, got {latent_output.shape}"
    assert reconstructed_output.shape == (batch_size, input_dim), \
        f"Reconstructed output shape mismatch. Expected {(batch_size, input_dim)}, got {reconstructed_output.shape}"
    print("Model forward pass shapes verified successfully.")
