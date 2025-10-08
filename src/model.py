"""
model.py

Defines the symmetric Encoder-Decoder network structure for the rMD Autoencoder,
incorporating the Swish activation function and a 3-dimensional latent space.
"""
import torch
import torch.nn as nn
from typing import Tuple

# Constants derived from paper specifications
LATENT_DIM = 3  # (M3)
INPUT_DIM = 9696 # (M5)


class Swish(nn.Module):
    """
    The Swish activation function: x * sigmoid(x).
    Used in all hidden layers (M2).
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class Encoder(nn.Module):
    """
    Compresses the 9696-dimensional input vector into the 3D latent space.
    The layer sizes are based on the *idea* of gradually shrinking layers (M1).
    The actual dimensions must be determined dynamically based on a reference structure.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        
        # Structure inspired by typical butterfly style (gradually shrinking)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4096),
            Swish(),
            nn.Linear(4096, 2048),
            Swish(),
            nn.Linear(2048, 1024),
            Swish(),
            nn.Linear(1024, 512),
            Swish(),
            nn.Linear(512, 128),
            Swish(),
            nn.Linear(128, latent_dim),  # Final latent space layer (M3)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """
    Decompresses the 3D latent space vector back to the 9696-dimensional structure vector.
    Mirrors the encoder architechture but in expanding order.
    """
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            Swish(),
            nn.Linear(128, 512),
            Swish(),
            nn.Linear(512, 1024),
            Swish(),
            nn.Linear(1024, 2048),
            Swish(),
            nn.Linear(2048, 4096),
            Swish(),
            nn.Linear(4096, output_dim), # Final reconstruction prediction layer
        )

    def forward(self, z):
        return self.layers(z)


class rMD_Autoencoder(nn.Module):
    """
    The complete Informed Autoencoder combining Encoder and Decoder.
    Note: Loss functions (Loss1, Loss2) will be handled externally in the training script,
    as Loss1 requires CV coordinates which are not part of the forward pass here.
    """
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compresses input structure to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs structure from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward pass: Reconstruction."""
        latent_vector = self.encode(x)
        reconstructed_x = self.decode(latent_vector)
        return reconstructed_x, latent_vector

# --- Unit Tests ---
def test_rmd_autoencoder():
    """Unit tests for the rMD Autoencoder architecture."""
    model = rMD_Autoencoder()
    
    # 1. Test dimensions (M3, M5)
    dummy_input = torch.randn(1, INPUT_DIM)
    
    reconstruction, latent = model.encode(dummy_input)
    
    assert reconstruction.shape[1] == INPUT_DIM, f"Reconstruction failed shape check: {reconstruction.shape[1]}"
    assert latent.shape[1] == LATENT_DIM, f"Latent space dimension failed: {latent.shape[1]}"
    
    # 2. Test full forward pass
    reconstruction_full, _ = model(dummy_input)
    assert reconstruction_full.shape[1] == INPUT_DIM, "Full pass reconstruction shape fails."
    
    # 3. Test Swish activation output shape
    swish = Swish()
    test_swish = swish(torch.randn(5))
    assert test_swish.shape == torch.Size([5]), "Swish activation shape test failed."
    
    print("Model architecture and dimension tests PASSED.")

if __name__ == '__main__':
    test_rmd_autoencoder()
