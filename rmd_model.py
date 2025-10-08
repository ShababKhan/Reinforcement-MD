import torch
import torch.nn as nn

# --- Constants imported from data_utils (to be shared across files) ---
# INPUT_DIM = 9696 (3232 atoms * 3 coords)
# CV_DIM = 3 (Latent Space dimension)
INPUT_DIM = 9696 
CV_DIM = 3

# Define the Swish Activation function as per the paper (Fig. S2 text)
class Swish(nn.Module):
    """
    Implements the Swish activation function: x * sigmoid(x).
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

# --- Architecture Definition (Based on Fig. S2) ---
# Layers are "gradually decreasing/increasing"
ENCODER_LAYERS = [
    (INPUT_DIM, 4096),
    (4096, 2048),
    (2048, 1024),
    (1024, 256),
    (256, CV_DIM) # Final layer to Latent Space
]

DECODER_LAYERS = [
    (CV_DIM, 256),  # Start from Latent Space
    (256, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, INPUT_DIM) # Final layer to Output (Prediction)
]


class Encoder(nn.Module):
    """
    T2.1: Implements the Encoder network.
    Compresses the flattened Cartesian coordinates (INPUT_DIM) into the 3D Latent Space (CV_DIM).
    Uses Fully Connected layers and Swish activation.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        layers = []
        
        for i, (in_dim, out_dim) in enumerate(ENCODER_LAYERS):
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Apply Swish activation to all hidden layers 
            # (i.e., not the last layer that outputs to CV_DIM)
            if i < len(ENCODER_LAYERS) - 1:
                layers.append(Swish())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights 
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming Uniform for ReLU-like activations (Swish)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param x: Input tensor of flattened Cartesian coordinates (N, INPUT_DIM).
        @return: Latent Space coordinates (N, CV_DIM).
        """
        return self.network(x)


class Decoder(nn.Module):
    """
    T2.2: Implements the Decoder network.
    Reconstructs the Cartesian coordinates (INPUT_DIM) from the 3D Latent Space (CV_DIM).
    Uses Fully Connected layers and Swish activation.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        layers = []
        
        for i, (in_dim, out_dim) in enumerate(DECODER_LAYERS):
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Apply Swish activation to all hidden layers 
            # (i.e., not the very last layer that outputs to INPUT_DIM)
            if i < len(DECODER_LAYERS) - 1:
                layers.append(Swish())
        
        self.network = nn.Sequential(*layers)

        # Initialize weights 
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Kaiming Uniform for ReLU-like activations (Swish)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param x: Input tensor of Latent Space coordinates (N, CV_DIM).
        @return: Predicted tensor of reconstructed Cartesian coordinates (N, INPUT_DIM).
        """
        return self.network(x)


class rMD_Network(nn.Module):
    """
    T2.3: The full Reinforced Molecular Dynamics Autoencoder Network.
    Combines the Encoder and Decoder with a single forward pass interface.
    """
    def __init__(self):
        super(rMD_Network, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the full autoencoder pass.

        @param x: Input tensor of flattened Cartesian coordinates (N, INPUT_DIM).
        @return: A tuple containing:
                 - LS_coords (torch.Tensor): Latent Space coordinates (N, CV_DIM).
                 - Pred_coords (torch.Tensor): Reconstructed Cartesian coordinates (N, INPUT_DIM).
        """
        # LS_coords is the output of the encoder (N, CV_DIM)
        LS_coords = self.encoder(x)
        
        # Pred_coords is the output of the decoder (N, INPUT_DIM)
        Pred_coords = self.decoder(LS_coords)
        
        return LS_coords, Pred_coords

if __name__ == '__main__':
    # --- Self-Test for T2.3 Acceptance Criteria ---
    
    # 1. Initialize the network and device
    device = torch.device("cpu")
    model = rMD_Network().to(device)
    
    # 2. Create dummy input data (e.g., a batch of 64 frames)
    BATCH_SIZE = 64
    dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)
    
    # 3. Perform a forward pass 
    print("Performing forward pass...")
    LS_output, Pred_output = model(dummy_input)
    
    # 4. Check layer dimensions 
    print(f"Input shape: {dummy_input.shape}")
    print(f"Latent Space Output (LS_coords) shape: {LS_output.shape}") 
    print(f"Prediction Output (Pred_coords) shape: {Pred_output.shape}")
    
    # Assertions for verification
    assert LS_output.shape == (BATCH_SIZE, CV_DIM)
    assert Pred_output.shape == (BATCH_SIZE, INPUT_DIM)
    print("Forward pass successful. Dimensions verified.")