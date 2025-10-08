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

def latent_loss(ls_coords: torch.Tensor, cv_coords: torch.Tensor) -> torch.Tensor:
    """
    Calculates the loss between latent space coordinates and collective variable coordinates.

    This function is designed to measure the correspondence between the low-dimensional
    latent space representation from the autoencoder and the biologically relevant
    collective variables. It computes an RMSD-like metric, analogous to how structural
    reconstruction is evaluated, but applied to the latent and CV spaces. This serves
    as Loss1 in the rMD informed autoencoder training.

    @param ls_coords: A PyTorch tensor of latent space coordinates.
                      Shape: (batch_size, latent_dim)
    @param cv_coords: A PyTorch tensor of collective variable coordinates.
                      Shape: (batch_size, cv_dim)

    @return: A scalar PyTorch tensor representing the RMSD-like loss between
             latent space and collective variable coordinates.
    @raises ValueError: If the input tensors ls_coords and cv_coords do not have
                        the same shape.
    """
    if ls_coords.shape != cv_coords.shape:
        raise ValueError("Input tensors ls_coords and cv_coords must have the same shape.")

    # Calculate squared difference
    squared_diff = (ls_coords - cv_coords)**2

    # Mean over all dimensions (latent/CV dimensions)
    mean_squared_diff = torch.mean(squared_diff)

    # Take the square root
    loss = torch.sqrt(mean_squared_diff)

    return loss
