"""
Autoencoder neural network for the rMD system.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """
    Encoder network for compressing protein structures into latent space.
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initialize the encoder network.
        
        Parameters:
        -----------
        input_dim : int
            Dimensionality of the input (flattened protein coordinates).
        hidden_dims : list
            List of hidden layer dimensions.
        latent_dim : int
            Dimensionality of the latent space.
        """
        super(Encoder, self).__init__()
        
        # Create list of linear layers with decreasing dimensions
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Swish())
            prev_dim = hidden_dim
        
        # Final layer to latent space does not have activation
        self.layers = nn.ModuleList(layers)
        self.latent_layer = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of flattened protein coordinates.
            
        Returns:
        --------
        torch.Tensor
            Encoded representation in latent space.
        """
        for layer in self.layers:
            x = layer(x)
        
        # No activation on the latent layer
        latent = self.latent_layer(x)
        
        return latent


class Decoder(nn.Module):
    """
    Decoder network for reconstructing protein structures from latent space.
    """
    
    def __init__(self, latent_dim, hidden_dims, output_dim):
        """
        Initialize the decoder network.
        
        Parameters:
        -----------
        latent_dim : int
            Dimensionality of the latent space.
        hidden_dims : list
            List of hidden layer dimensions (in reverse order compared to encoder).
        output_dim : int
            Dimensionality of the output (flattened protein coordinates).
        """
        super(Decoder, self).__init__()
        
        # Create list of linear layers with increasing dimensions
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Swish())
            prev_dim = hidden_dim
        
        # Final layer to output space does not have activation
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass through the decoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor from latent space.
            
        Returns:
        --------
        torch.Tensor
            Reconstructed protein coordinates.
        """
        for layer in self.layers:
            x = layer(x)
        
        # No activation on the output layer
        output = self.output_layer(x)
        
        return output


class InformedAutoencoder(nn.Module):
    """
    Autoencoder with dual loss function for rMD.
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """
        Initialize the autoencoder network.
        
        Parameters:
        -----------
        input_dim : int
            Dimensionality of the input (flattened protein coordinates).
        hidden_dims : list
            List of hidden layer dimensions for the encoder.
            Will be reversed for the decoder.
        latent_dim : int
            Dimensionality of the latent space (should match CV dimension).
        """
        super(InformedAutoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
        
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of flattened protein coordinates.
            
        Returns:
        --------
        tuple
            Tuple containing:
            - latent: Encoded representation in latent space.
            - output: Reconstructed protein coordinates.
        """
        latent = self.encoder(x)
        output = self.decoder(latent)
        
        return latent, output
    
    def encode(self, x):
        """
        Encode protein coordinates to latent space.
        
        Parameters:
        -----------
        x : torch.Tensor or numpy.ndarray
            Input tensor or array of flattened protein coordinates.
            
        Returns:
        --------
        torch.Tensor
            Encoded representation in latent space.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            
        with torch.no_grad():
            latent = self.encoder(x)
            
        return latent
    
    def decode(self, z):
        """
        Decode from latent space to protein coordinates.
        
        Parameters:
        -----------
        z : torch.Tensor or numpy.ndarray
            Input tensor or array in latent space.
            
        Returns:
        --------
        torch.Tensor
            Reconstructed protein coordinates.
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
            
        with torch.no_grad():
            output = self.decoder(z)
            
        return output


class RMDLoss(nn.Module):
    """
    Dual loss function for rMD, combining Loss1 (latent to CV) and Loss2 (reconstruction).
    """
    
    def __init__(self, loss1_weight=1.0, loss2_weight=1.0):
        """
        Initialize the loss function.
        
        Parameters:
        -----------
        loss1_weight : float
            Weight for Loss1 (latent to CV mapping).
        loss2_weight : float
            Weight for Loss2 (reconstruction error).
        """
        super(RMDLoss, self).__init__()
        self.loss1_weight = loss1_weight
        self.loss2_weight = loss2_weight
        
    def forward(self, latent, cv_coords, reconstructed, original):
        """
        Calculate the combined loss.
        
        Parameters:
        -----------
        latent : torch.Tensor
            Encoded latent space coordinates.
        cv_coords : torch.Tensor
            Target collective variable coordinates.
        reconstructed : torch.Tensor
            Reconstructed protein coordinates.
        original : torch.Tensor
            Original protein coordinates.
            
        Returns:
        --------
        tuple
            Tuple containing:
            - total_loss: Weighted sum of Loss1 and Loss2.
            - loss1: MSE between latent and CV coordinates.
            - loss2: MSE between original and reconstructed coordinates.
        """
        # Loss1: Mean squared error between latent space and CV space
        loss1 = F.mse_loss(latent, cv_coords)
        
        # Loss2: Mean squared error for reconstruction
        loss2 = F.mse_loss(reconstructed, original)
        
        # Weighted combination
        total_loss = self.loss1_weight * loss1 + self.loss2_weight * loss2
        
        return total_loss, loss1, loss2


def build_autoencoder(n_atoms, hidden_layer_config, latent_dim=3):
    """
    Build an autoencoder with the specified configuration.
    
    Parameters:
    -----------
    n_atoms : int
        Number of atoms in the protein structure.
    hidden_layer_config : list
        List of hidden layer dimensions for the encoder.
    latent_dim : int
        Dimensionality of the latent space.
        
    Returns:
    --------
    InformedAutoencoder
        Configured autoencoder model.
    """
    # Input dimension is 3 * n_atoms for flattened coordinates
    input_dim = 3 * n_atoms
    
    # Build the autoencoder
    model = InformedAutoencoder(input_dim, hidden_layer_config, latent_dim)
    
    return model


def rmsd_loss(reconstructed, original, batch_size=None):
    """
    Calculate RMSD between original and reconstructed structures.
    
    Parameters:
    -----------
    reconstructed : torch.Tensor
        Reconstructed coordinates with shape (batch_size, n_atoms * 3).
    original : torch.Tensor
        Original coordinates with shape (batch_size, n_atoms * 3).
    batch_size : int, optional
        Batch size. If None, inferred from tensors.
        
    Returns:
    --------
    torch.Tensor
        Mean RMSD across the batch.
    """
    if batch_size is None:
        batch_size = original.shape[0]
        
    # Reshape to (batch_size, n_atoms, 3)
    n_atoms = original.shape[1] // 3
    
    original_3d = original.view(batch_size, n_atoms, 3)
    reconstructed_3d = reconstructed.view(batch_size, n_atoms, 3)
    
    # Calculate squared differences
    squared_diff = torch.sum((original_3d - reconstructed_3d) ** 2, dim=2)
    
    # Calculate RMSD for each structure in the batch
    rmsd = torch.sqrt(torch.mean(squared_diff, dim=1))
    
    # Return mean RMSD across the batch
    return torch.mean(rmsd)