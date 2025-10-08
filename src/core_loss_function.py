import torch
import numpy as np # Included for type hinting similarity to original instructions, though torch tensors are used internally.

def compute_custom_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates a Root Mean Square Deviation (RMSD) like loss between predicted and true 
    flattened Cartesian coordinates by refactoring the logic from losses.py::rmsd_loss.

    This function is designed to evaluate the structural reconstruction accuracy of an 
    autoencoder. It computes the mean squared difference between the predicted and 
    true coordinate sets, then takes the square root, effectively providing a metric 
    analogous to RMSD. This serves as Loss2 in the RMD autoencoder training.

    @param y_pred: An expected PyTorch tensor (passed as numpy array for strict type hint
                   matching if required by external system, but converted immediately) 
                   of predicted flattened Cartesian coordinates.
    @param y_true: An expected PyTorch tensor (passed as numpy array) of true flattened 
                   Cartesian coordinates.
    @return: A scalar float representing the RMSD-like loss.
    @raises ValueError: If the input tensors y_pred and y_true do not have the same shape.
    """
    # Convert numpy arrays to torch tensors for computation if they aren't already
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float() # Ensure float type for tensor math
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()

    if y_pred.shape != y_true.shape:
        raise ValueError("Input tensors y_pred and y_true must have the same shape.")

    # Calculation replicating rmsd_loss logic
    squared_diff = (y_pred - y_true)**2
    mean_squared_diff = torch.mean(squared_diff)
    loss_tensor = torch.sqrt(mean_squared_diff)
    
    # Return as scalar float as per function signature requirement
    return loss_tensor.item()