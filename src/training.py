"""
training.py (Sprint 2.2, 2.3)

Contains the main training loop for the Informed rMD Autoencoder, simultaneously optimizing
reconstruction loss (Loss2) and latent space correlation loss (Loss1).
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple

from src.model import rMD_Autoencoder, LATENT_DIM, INPUT_DIM
from src.data_loader import load_and_split_data, INPUT_VECTOR_SIZE
from src.cv_calculator import calculate_cbn_cvs

# --- Training Hyperparameters based on Paper (M9, M10) ---
N_EPOCHS = 10000     # Training rounds (referred to as epochs here for PyTorch convention)
BATCH_SIZE = 64      # (M10)
LEARNING_RATE = 1e-4 # Assumed starting LR for Adam
# Optimizer: Adams (M9) -> We use Adam, a common interpretation in modern ML
OPTIMIZER_TYPE = optim.Adam 
# Loss Weights: The paper mentions a 'weighted sum', but specifies no weight ratio.
# We start with equal weighting (w1=1.0, w2=1.0) and prioritize Loss1 slightly 
# to ensure physics infusion as suggested by the target values (Loss1 ~ 1.0, Loss2 ~ 1.6).
W_LOSS1 = 1.5  # Weight for Latent Loss (should guide LS structure)
W_LOSS2 = 1.0  # Weight for Prediction Loss (should guide structure fidelity)

# --- Mock Setup Files (Must be provided externally in a real run) ---
MOCK_TOPOLOGY = "data/crbn_topology.pdb"
MOCK_TRAJECTORY_LIST = [f"data/frame_{i}.dcd" for i in range(8000 + 2000)] # Mocking 10k frames
REF_PDB = "data/crbn_reference_frame.pdb"

# --- Loss Functions ---

def loss2_reconstruction(reconstructed_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
    """
    Loss2: Average Root Mean Square Distance (RMSD) approximation via MAE.
    Minimizes structural difference between input and output.
    """
    # Using Mean Absolute Error (MAE) as an approximation for L2 (RMSD) loss mentioned in text (M7)
    return F.l1_loss(reconstructed_x, target_x)


def loss1_latent_physics(latent_z: torch.Tensor, target_cvs: torch.Tensor) -> torch.Tensor:
    """
    Loss1: Enforces a correspondence between Latent Space (LS) coordinates and CV coordinates.
    
    NOTE: Since CV calculation requires non-tensor data (e.g., MDAnalysis objects), 
    this function relies on pre-computed CVs being passed as a tensor target.
    """
    # target_cvs must be a batch of 3 CV values for each latent vector in the batch
    return F.mse_loss(latent_z, target_cvs) # MSE is often used for spatial distance matching


# --- Training Logic ---

def train_one_epoch_rMD(
    model: rMD_Autoencoder, 
    data_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> Tuple[float, float]:
    """Performs one training pass over the dataset."""
    model.train()
    total_loss2, total_loss1 = 0.0, 0.0
    
    for batch_idx, input_x in enumerate(data_loader):
        input_x = input_x.to(device)
        
        # 1. Forward pass
        reconstructed_x, latent_z = model(input_x)
        
        # 2. Calculate Loss 2 (Reconstruction)
        loss2 = loss2_reconstruction(reconstructed_x, input_x)
        
        # 3. Calculate Loss 1 (Physics Infusion) - MOCK/PLACEHOLDER
        # CRITICAL HACK: In a real environment, we cannot run CV calculation (which needs atom coords
        # and MDAnalysis) inside the tight PyTorch loop. We must generate mock CV targets here
        # that match the batch size of the input.
        
        batch_size = input_x.size(0)
        # Mock target CVs: 3 dimensions, batch_size samples. These must be replaced. (MOCK FOR SPRINT 2)
        # In a full implementation, this would involve iterating through input_x, calculating CVs, and converting to tensor.
        # For this structural check, we generate random vectors that *look* like CVs in the expected range.
        target_cvs = torch.randn(batch_size, LATENT_DIM).to(device) * 5.0
        
        loss1 = loss1_latent_physics(latent_z, target_cvs)
        
        # 4. Combined Loss and Backpropagation (M1)
        combined_loss = (W_LOSS2 * loss2) + (W_LOSS1 * loss1)
        
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        
    avg_loss1 = total_loss1 / len(data_loader)
    avg_loss2 = total_loss2 / len(data_loader)
    return avg_loss1, avg_loss2


def train_rmd_model(epochs: int, train_loader: DataLoader, val_loader: DataLoader) -> rMD_Autoencoder:
    """Main training scheduler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = rMD_Autoencoder().to(device)
    optimizer = OPTIMIZER_TYPE(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting rMD Training for {epochs} rounds (10000 specified by paper).")
    
    best_loss1, best_loss2 = float('inf'), float('inf')

    for epoch in range(1, epochs + 1):
        
        # Training step
        train_l1, train_l2 = train_one_epoch_rMD(model, train_loader, optimizer, device)
        
        # Validation step (Simplified: validation loop here would ideally check CV fidelity too)
        # For this step, we skip a full validation loop for simplicity and focus on tracking training loss descent.
        
        # Check convergence against paper targets (M7, M8)
        if epoch % 1000 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | L2 (Pred): {train_l2:.4f} Å | L1 (Latent): {train_l1:.4f} Å")
            
            # Update best values (M7, M8 tracking)
            if train_l2 < best_loss2:
                best_loss2 = train_l2
            if train_l1 < best_loss1:
                best_loss1 = train_l1
                
    print("\n--- Training Finished ---")
    print(f"Final best Loss2 (Target ~1.6 Å): {best_loss2:.3f} Å")
    print(f"Final best Loss1 (Target ~1.0 Å): {best_loss1:.3f} Å")
    
    return model


if __name__ == '__main__':
    # Execute the data loading simulation first
    try:
        # NOTE: The mock data loader simulates the loading of 10,000 frames (8k/2k split)
        train_loader, val_loader, _, _ = load_and_split_data(MOCK_TOPOLOGY, MOCK_TRAJECTORY_LIST, REF_PDB)
        
        # Since Loss1 requires real CVs, this training run will only verify gradient flow 
        # and convergence properties, not true physical mapping.
        trained_model = train_rmd_model(epochs=1000, train_loader=train_loader, val_loader=val_loader)
        
    except FileNotFoundError:
        print("\n[ERROR] Cannot run training script: Mock data files (PDB/DCD) for data_loader setup are missing.")
    except Exception as e:
        print(f"\n[GENERAL ERROR DURING TRAINING SIMULATION]: {e}")

