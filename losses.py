import torch

def rmsd_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates a Root Mean Square Deviation (RMSD) like loss between predicted and true
    flattened Cartesian coordinates.

    This function is designed to evaluate the structural reconstruction accuracy of an
    autoencoder. It computes the mean squared difference between the predicted and
    true coordinate sets, then takes the square root, effectively providing a metric
    analogous to RMSD. This serves as Loss2 in the rMD autoencoder training.

    @param y_pred: A PyTorch tensor of predicted flattened Cartesian coordinates.
                   Shape: (batch_size, num_atoms * 3)
    @param y_true: A PyTorch tensor of true flattened Cartesian coordinates.
                   Shape: (batch_size, num_atoms * 3)

    @return: A scalar PyTorch tensor representing the RMSD-like loss.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("Input tensors y_pred and y_true must have the same shape.")

    # Calculate squared difference
    squared_diff = (y_pred - y_true)**2

    # Mean over all dimensions (atoms and coordinates)
    mean_squared_diff = torch.mean(squared_diff)

    # Take the square root
    loss = torch.sqrt(mean_squared_diff)

    return loss
