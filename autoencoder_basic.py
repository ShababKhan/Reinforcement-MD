import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Constants based on Paper Description ---
# The paper states the input vector length is 9696 (for CRBN heavy atoms).
# We will use a placeholder size for demonstration.
INPUT_FEATURES = 9696
LATENT_DIM = 3  # Paper uses 3 dimensions for the LS in the CRBN example

class CRBNAutoencoder(nn.Module):
    """
    A basic Encoder-Decoder architecture mimicking the structure described
    in the rMD paper (Fig. 2 and Fig. S2).

    The network compresses input structure coordinates into a low-dimensional
    latent space (LS) and then reconstructs them. Implements the structural
    reconstruction path (Encoder -> LS -> Decoder).
    """
    def __init__(self, input_dim: int, latent_dim: int):
        """
        @param input_dim: The total number of flattened Cartesian coordinates.
        @param latent_dim: The dimension of the latent space.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # --- Encoder Layers (Gradually Shrinking) ---
        # Based on the paper's description of gradually shrinking layers.
        # We choose arbitrary dimension sizes common in autoencoders leading to LS=3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.Swish(),  # Using Swish activation as mentioned in the text
            nn.Linear(4096, 2048),
            nn.Swish(),
            nn.Linear(2048, 1024),
            nn.Swish(),
            nn.Linear(1024, 512),
            nn.Swish(),
            nn.Linear(512, latent_dim)  # Latent Space (LS)
        )

        # --- Decoder Layers (Gradually Expanding) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.Swish(),
            nn.Linear(512, 1024),
            nn.Swish(),
            nn.Linear(1024, 2048),
            nn.Swish(),
            nn.Linear(2048, 4096),
            nn.Swish(),
            nn.Linear(4096, input_dim), # Reconstruction layer
            # NOTE: The paper does not explicitly mention the final activation
            # for reconstruction (as coordinates can be positive/negative),
            # so we omit a final linear activation unless necessary for shape context.
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compresses input structure into latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs structure from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder and decoder.
        @param x: Input structure coordinates.
        @return: A tuple (reconstructed_x, latent_vector_z)
        """
        z = self.encode(x)
        reconstructed_x = self.decode(z)
        return reconstructed_x, z


def calculate_rmsd_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Loss2 (predLoss): The average Root Mean Square Distance (RMSD)
    between predicted and target structures.

    RMSD = sqrt( mean( (P_i - T_i)^2 ) )
    The loss function typically minimizes the squared term (Mean Squared Error - MSE)
    which is proportional to the RMSD. We return the MSE as the standard reconstruction loss.

    @param predicted: The reconstructed structure coordinates (N_batch, N_features)
    @param target: The original input structure coordinates (N_batch, N_features)
    @return: The average Mean Squared Error (MSE) loss across the batch.
    """
    # Calculate squared difference element-wise
    squared_diff = (predicted - target) ** 2
    
    # Calculate the mean over all elements (features and batch dimensions)
    mse_loss = torch.mean(squared_diff)
    
    # Note: The paper minimizes the average RMSD, which is equivalent to
    # minimizing the MSE loss function used in standard training loops.
    return mse_loss


if __name__ == '__main__':
    print("--- Basic Autoencoder Initialization Test ---")
    # 1. Initialize Model
    model = CRBNAutoencoder(input_dim=INPUT_FEATURES, latent_dim=LATENT_DIM)
    print(f"Model initialized. Input Dim: {INPUT_FEATURES}, Latent Dim: {LATENT_DIM}")

    # 2. Create Dummy Data (Batch size = 2 structures, 9696 features each)
    batch_size = 2
    dummy_input = torch.randn(batch_size, INPUT_FEATURES)

    # 3. Forward Pass
    reconstructed_output, latent_vector = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Latent vector shape (z): {latent_vector.shape}")
    print(f"Reconstruction shape: {reconstructed_output.shape}")

    # 4. Calculate Loss2 (predLoss)
    loss2 = calculate_rmsd_loss(reconstructed_output, dummy_input)
    
    # To verify the RMSD calculation, we manually calculate the true RMSD
    # for one sample and compare the scale of the MSE loss.
    mse_numpy = np.mean((dummy_input.numpy() - reconstructed_output.detach().numpy())**2)
    
    print(f"\nCalculated Loss2 (MSE): {loss2.item():.6f}")
    print(f"Manual Check (MSE):    {mse_numpy:.6f}")
    assert np.isclose(loss2.item(), mse_numpy), "PyTorch loss calculation mismatch with manual NumPy check."
    
    # Assert features are consistent
    assert latent_vector.shape == (batch_size, LATENT_DIM)
    assert reconstructed_output.shape == dummy_input.shape
    
    print("\n--- Basic Autoencoder test successful. ---")

