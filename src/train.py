import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from src.model_architecture import InformedAutoencoder
from src.data_utils import load_r_batch_data
from src.losses import DualLoss

# Hyperparameters derived from the paper (S1.M13, S1.M6)
N_ROUNDS = 10000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  # Standard starting LR for Adam
L1_ALPHA = 0.1       # Weighted sum factor, derived from S1.M6 (alpha=0.1 based on loss ratio targets)

def train_r_model(num_epochs: int = N_ROUNDS) -> Tuple[InformedAutoencoder, float, float]:
    """
    Trains the Informed Autoencoder model using the dual-loss function (S1.T6).
    
    The method simulates the data loading, network initialization, and 
    training loop described in the paper (10k rounds, Batch 64, Adams Optimizer).

    @param num_epochs: The number of training rounds/iterations to perform.
    @return: A tuple containing the trained model, final Loss1, and final Loss2.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data (Using mock data for initiation, matching S1.T1 output)
    X_data, Y_cv_data = load_r_batch_data()
    
    # Split data: 8000 train, 2000 validation (Paper S1.M11)
    X_train, X_val = X_data[:8000], X_data[8000:]
    Y_cv_train, Y_cv_val = Y_cv_data[:8000], Y_cv_data[8000:]
    
    train_dataset = TensorDataset(
        torch.tensor(X_train).to(device), 
        torch.tensor(X_train).to(device), # Target is the input itself for Loss 2
        torch.tensor(Y_cv_train).to(device)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val).to(device), 
        torch.tensor(X_val).to(device), 
        torch.tensor(Y_cv_val).to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model, Loss, and Optimizer
    model = InformedAutoencoder().to(device)
    criterion = DualLoss(alpha=L1_ALPHA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training for {num_epochs} rounds...")

    best_l2 = float('inf')

    # 4. Training Loop (S1.T6)
    for epoch in range(1, num_epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        
        for X_batch, Target_batch, Y_cv_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            prediction, ls_coords = model(X_batch)
            
            # Calculate losses
            total_loss, loss1, loss2 = criterion(prediction, Target_batch, ls_coords, Y_cv_batch)
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            running_loss1 += loss1.item() * X_batch.size(0)
            running_loss2 += loss2.item() * X_batch.size(0)

        epoch_loss1 = running_loss1 / len(train_loader.dataset)
        epoch_loss2 = running_loss2 / len(train_loader.dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss1_sum = 0.0
        val_loss2_sum = 0.0
        
        with torch.no_grad():
            for X_val_batch, Target_val_batch, Y_cv_val_batch in val_loader:
                prediction_val, ls_coords_val = model(X_val_batch)
                _, loss1_val, loss2_val = criterion(prediction_val, Target_val_batch, ls_coords_val, Y_cv_val_batch)
                
                val_loss1_sum += loss1_val.item() * X_val_batch.size(0)
                val_loss2_sum += loss2_val.item() * X_val_batch.size(0)

        val_loss1 = val_loss1_sum / len(val_loader.dataset)
        val_loss2 = val_loss2_sum / len(val_loader.dataset)

        if epoch % 1000 == 0 or epoch == num_epochs:
            print(f"Epoch [{epoch}/{num_epochs}] | Train L1: {epoch_loss1:.4f}, Train L2: {epoch_loss2:.4f} | "
                  f"Val L1: {val_loss1:.4f}, Val L2: {val_loss2:.4f}")
            
            # Saving the state dictionary corresponding to the best L2 RMSD achieved so far
            if val_loss2 < best_l2:
                best_l2 = val_loss2
                torch.save(model.state_dict(), "r_model_best_l2.pth")
                print(" -> Saved checkpoint: r_model_best_l2.pth")
    
    # Load best model state for final return values (simulating final check)
    model.load_state_dict(torch.load("r_model_best_l2.pth"))

    print("\nTraining Complete.")
    # The paper reports final losses: Loss1 approx 1.0, Loss2 approx 1.6
    return model, val_loss1, val_loss2

if __name__ == '__main__':
    # NOTE: This simulation will use random initialization and mock data, 
    # so the final loss values will *not* perfectly match the paper's 1.0 and 1.6 targets, 
    # but the process flow is validated.
    
    trained_model, final_l1, final_l2 = train_r_model(num_epochs=1000) # Use fewer epochs for quick test
    
    print(f"*** Final Validation Loss Results (Mock Data) ***")
    print(f"Final Loss 1 (Latent): {final_l1:.4f}")
    print(f"Final Loss 2 (Reconstruction/RMSD): {final_l2:.4f} (Target: ~1.6)")