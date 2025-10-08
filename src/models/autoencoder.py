import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Applies the Swish activation function element-wise.
    Swish(x) = x * sigmoid(x)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param x: The input tensor.
        @type x: torch.Tensor
        @return: The output tensor after applying Swish activation.
        @rtype: torch.Tensor
        """
        return x * torch.sigmoid(x)


class BasicAutoencoder(nn.Module):
    """
    Implements the basic autoencoder architecture for protein structure prediction
    as described in the paper's Figure 2 and Figure S2, without the additional
    latent space loss function (Loss1).

    The encoder compresses flattened Cartesian coordinates into a low-dimensional
    latent space (3 dimensions in this case). The decoder then reconstructs the
    original Cartesian representation from this latent space.

    The architecture consists of fully connected layers with gradually shrinking
    and expanding neuron counts, using the Swish activation function between layers.

    @param input_dim: The dimension of the flattened input Cartesian coordinates
                      (e.g., 9696 for CRBN heavy atoms).
    @type input_dim: int
    @param latent_dim: The dimension of the latent space (e.g., 3 for CVs).
    @type latent_dim: int
    """
    def __init__(self, input_dim: int = 9696, latent_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 5000),
            Swish(),
            nn.Linear(5000, 1000),
            Swish(),
            nn.Linear(1000, 500),
            Swish(),
            nn.Linear(500, latent_dim)  # Latent space
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500),
            Swish(),
            nn.Linear(500, 1000),
            Swish(),
            nn.Linear(1000, 5000),
            Swish(),
            nn.Linear(5000, input_dim)  # Output layer
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the autoencoder.

        @param x: The input tensor of flattened Cartesian coordinates
                  of shape (batch_size, input_dim).
        @type x: torch.Tensor
        @return: A tuple containing:
                 - latent_representation (torch.Tensor): The output of the encoder
                                                         (latent space coordinates).
                 - reconstructed_x (torch.Tensor): The output of the decoder
                                                   (reconstructed input coordinates).
        @rtype: tuple[torch.Tensor, torch.Tensor]
        """
        latent_representation = self.encoder(x)
        reconstructed_x = self.decoder(latent_representation)
        return latent_representation, reconstructed_x
