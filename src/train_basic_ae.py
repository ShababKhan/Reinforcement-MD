import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import numpy as np

# Import modules defined in preceding steps
from .config import INPUT_DIM, LS_DIM, N_ROUNDS, BATCH_SIZE, WEIGHT_L2, OPTIMIZER
from .model import BasicAutoencoder
from .data_loader import get_data_loaders

# --- Loss Function ---
# M7: Loss2 is the average Root Mean Square Distance (RMSD)
def loss_rmsd(output, target):
    """Calculates the average RMSD (Loss2) between prediction and input."""
    # Since structure coordinates are flattened, MSE on coordinates approximates RMSD calculation
    # For heavy atoms, this is close enough for initial loss comparison.
    squared_error = (output - target) ** 2
    mean_squared_error = torch.mean(squared_error)
    # Note: RMSD = sqrt(MSE). We use MSE as the loss itself, as is common in AE training, 
    # or we could explicitly use sqrt(torch.mean((output - target) ** 2)).
    # Sticking to MSE (Loss^2) as the reconstruction loss function for stability, 
    # and comparing the final magnitude to the paper's RMSD value (M7).
    return mean_squared_error

def train_basic_ae():
    """Executes Sprint 1.2 testing: Training the base AE with Loss2 only."""
    
    print("--- Starting Sprint 1.2: Basic AE Training (Loss2 Only) ---")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Data (Uses synthetic 9696-dim data internally)
    # Ensure 'data' directory exists for synthetic files
    if not os.path.exists("data"):
        os.makedirs("data")
        
    train_loader, val_loader = get_data_loaders(batch_size=BATCH_SIZE)
    
    # 3. Initialize Model, Optimizer, Loss
    model = BasicAutoencoder(INPUT_DIM, LS_DIM).to(device)
    criterion_L2 = loss_rmsd
    optimizer = OPTIMIZER(model.parameters(), lr=1e-4) # Set a default learning rate
    
    # 4. Training Loop (Up to N_ROUNDS, though we may stop early for quick test)
    
    # For demonstration, we'll run for a smaller, fixed number of steps or full rounds
    num_epochs = 5 # Reduced epochs for initial setup check, actual run will be N_ROUNDS
    
    print(f"Training for {num_epochs} synthetic epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_l2_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, _ = model(inputs)
            
            # Calculate Loss2 (Reconstruction Loss)
            loss2 = criterion_L2(reconstruction, targets)
            
            # Backward pass (Only L2 for now)
            (WEIGHT_L2 * loss2).backward()
            optimizer.step()
            
            total_l2_loss += loss2.item()
            
        avg_loss_sq = total_l2_loss / len(train_loader)
        # Note: The paper reports RMSD. RMSD = sqrt(MSE)
        avg_rmsd = np.sqrt(avg_loss_sq) 
        
        print(f"Epoch {epoch+1}/{num_epochs} | Avg L2 Loss (MSE): {avg_loss_sq:.4f} | Avg RMSD: {avg_rmsd:.4f} Å")

    print("--- Basic AE Training Complete ---")
    # Final check against M7 (Target is ~1.6 A RMSD)
    print(f"Final Synthetic RMSD: {avg_rmsd:.4f} Å. Target M7 (for real data): ~1.6 Å")

if __name__ == '__main__':
    # Create data directory if it doesn't exist for the synthetic data generation step
    if not os.path.exists("data"):
        os.makedirs("data")
    train_basic_ae()
