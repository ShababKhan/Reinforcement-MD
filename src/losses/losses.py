import torch
import torch.nn.functional as F

def rmsd_loss(y_pred, y_true):
    """
    Calculates a Root Mean Square Deviation (RMSD) like loss between predicted and true coordinates.

    This function computes the square root of the mean of the squared differences
    between corresponding elements of two input tensors. It is used to evaluate
    the structural reconstruction error in the autoencoder.

    @param y_pred (torch.Tensor): The predicted coordinates tensor.
                                  Shape: (batch_size, num_atoms * 3)
    @param y_true (torch.Tensor): The true coordinates tensor.
                                  Shape: (batch_size, num_atoms * 3)

    @return (torch.Tensor): A scalar tensor representing the RMSD-like loss.

    @raises ValueError: If y_pred and y_true do not have the same shape.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("Input tensors y_pred and y_true must have the same shape.")
    squared_diff = (y_pred - y_true).pow(2)
    loss = torch.sqrt(torch.mean(squared_diff))
    return loss

def latent_loss(ls_coords, cv_coords):
    """
    Calculates a loss to enforce correspondence between latent space coordinates
    and collective variable coordinates.

    This function computes an RMSD-like distance between the latent space
    representation and the target collective variable representation.
    This is used to "infuse" the autoencoder's latent space with physical meaning.

    @param ls_coords (torch.Tensor): The latent space coordinates tensor.
                                     Shape: (batch_size, latent_dim)
    @param cv_coords (torch.Tensor): The collective variable coordinates tensor.
                                     Shape: (batch_size, cv_dim)

    @return (torch.Tensor): A scalar tensor representing the latent space correspondence loss.

    @raises ValueError: If ls_coords and cv_coords do not have the same shape.
    """
    if ls_coords.shape != cv_coords.shape:
        raise ValueError("Input tensors ls_coords and cv_coords must have the same shape.")
    squared_diff = (ls_coords - cv_coords).pow(2)
    loss = torch.sqrt(torch.mean(squared_diff))
    return loss
