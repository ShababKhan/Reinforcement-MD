import torch

def rmsd_loss(input_coords: torch.Tensor, output_coords: torch.Tensor) -> torch.Tensor:
    """
    Calculates a Root Mean Square Deviation (RMSD) like loss between two sets
    of flattened Cartesian coordinates.

    This function computes the Euclidean distance between corresponding points
    in the input and output coordinate tensors, squares these distances,
    averages them, and then takes the square root. This serves as a
    reconstruction loss (Loss2 as described in the paper) for the autoencoder.

    Note: This simplified RMSD calculation assumes that the input and output
    structures are already aligned (superposed), as suggested by the paper's
    methodology (superposition to a single frame prior to training).
    It does not perform explicit superposition.

    @param input_coords: A PyTorch tensor of shape (batch_size, num_atoms * 3)
                         representing the reference flattened Cartesian coordinates.
    @type input_coords: torch.Tensor
    @param output_coords: A PyTorch tensor of shape (batch_size, num_atoms * 3)
                          representing the predicted flattened Cartesian coordinates.
    @type output_coords: torch.Tensor

    @return: A scalar PyTorch tensor representing the average RMSD loss over the batch.
    @rtype: torch.Tensor

    @raises ValueError: If input_coords and output_coords have different shapes.
    """
    if input_coords.shape != output_coords.shape:
        raise ValueError("Input and output coordinates must have the same shape.")

    # Calculate squared differences
    squared_diff = (input_coords - output_coords) ** 2

    # Sum over all coordinate dimensions for each sample, then average over samples
    # The paper mentions 'average root mean square distance (RMSD) between inputs and their predicted outputs'
    # For flattened coordinates, we sum over all 9696 dimensions and then average over the batch.
    loss = torch.sqrt(torch.mean(torch.sum(squared_diff, dim=-1)))

    return loss
