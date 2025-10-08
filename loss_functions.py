import torch
import torch.nn as nn
from typing import Tuple

# Loss function constants
DEFAULT_W1 = 1.0  # Weight for Loss1 (Latent Loss)
DEFAULT_W2 = 1.0  # Weight for Loss2 (Reconstruction Loss)

class DualLoss(nn.Module):
    """
    Implements the weighted dual-loss function required by the rMD methodology:
    Total Loss = (W1 * Loss1) + (W2 * Loss2)
    
    Loss1 (Latent Loss): Penalizes mismatch between Latent Space (LS) coordinates 
                         and Collective Variable (CV) targets. (Physics Infusion)
    Loss2 (Reconstruction Loss): Penalizes mismatch between reconstructed coordinates 
                                 and original input coordinates (Autoencoder Fidelity).

    The network is dual-optimized to ensure the latent space is physically meaningful 
    while preserving structural reconstruction quality.
    """
    def __init__(self, w1: float = DEFAULT_W1, w2: float = DEFAULT_W2):
        """
        Initializes the loss function layers and weights.
        
        @param w1: Weight for Loss1 (Latent Loss).
        @param w2: Weight for Loss2 (Reconstruction Loss).
        """
        super(DualLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        
        # T3.1: Loss 1 (Latent Loss) - Penalizes difference between LS and CVs. 
        # Using MSE (L2) as a standard choice for low dimensions.
        self.loss1_func = nn.MSELoss(reduction='mean') 
        
        # T3.2: Loss 2 (Reconstruction Loss) - Penalizes difference between 
        # predicted and target coordinates. Using L1 Loss (MAE) as a robust proxy 
        # for RMSD/L1 loss in high dimensions, as done in many autoencoder papers.
        self.loss2_func = nn.L1Loss(reduction='mean') 

    def forward(self, 
                ls_coords: torch.Tensor, 
                pred_coords: torch.Tensor, 
                target_cvs: torch.Tensor, 
                target_coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the weighted total loss.

        @param ls_coords: Output from the Encoder (Latent Space coordinates).
        @param pred_coords: Output from the Decoder (Reconstructed coordinates).
        @param target_cvs: Collective Variable targets (Ground Truth).
        @param target_coords: Original Cartesian input coordinates (Ground Truth).
        @return: A tuple (total_loss, loss1, loss2).
        """
        # T3.1 Implementation: Loss1 (Latent Loss) 
        loss1 = self.loss1_func(ls_coords, target_cvs)
        
        # T3.2 Implementation: Loss2 (Reconstruction Loss)
        loss2 = self.loss2_func(pred_coords, target_coords)
        
        # T3.3 Implementation: Weighted Total Loss
        total_loss = (self.w1 * loss1) + (self.w2 * loss2)
        
        return total_loss, loss1, loss2

    def get_weights(self) -> Tuple[float, float]:
        """Returns the current loss weights (W1, W2)."""
        return self.w1, self.w2

    @staticmethod
    def calculate_rmsd(pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Root Mean Square Deviation (RMSD) between predicted and 
        target coordinates (Loss2 metric, calculated as RMSD for reporting).
        
        RMSD = sqrt( sum((P_i - T_i)^2) / N_coords )
        
        @param pred_coords: Predicted Cartesian coordinates (N, INPUT_DIM).
        @param target_coords: Target Cartesian coordinates (N, INPUT_DIM).
        @return: Mean RMSD over the batch.
        """
        n_coords = target_coords.shape[1]
        
        # RMSD for each frame in the batch
        rmsd_per_frame = torch.sqrt(torch.sum((pred_coords - target_coords)**2, dim=1) / n_coords)
        
        # Mean RMSD over the batch
        mean_rmsd = torch.mean(rmsd_per_frame)
        
        return mean_rmsd

