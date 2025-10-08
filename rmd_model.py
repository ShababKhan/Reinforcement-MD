import torch
import torch.nn as nn

class RMDModel(nn.Module):
    """
    Initial structure for the Reinforcement Molecular Dynamics (RMD) Agent.
    This model takes the featurized state vector (from data_utils.py) and 
    outputs a vector corresponding to the actions (e.g., forces or Q-values) to take.
    
    This is a preliminary structure for US-1.3 and will be replaced by the full 
    Autoencoder architecture (T2.1, T2.2, T2.3) in the next sprint.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        :param input_dim: The size of the featurized state vector (N_atoms * 3).
        :param hidden_dim: The size of the hidden layers.
        """
        super(RMDModel, self).__init__()
        
        # Output dimension is the same as the input dimension, as the agent
        # predicts forces/displacements for every x, y, z coordinate, or a Q-value 
        # map of equal size, depending on the RL approach.
        self.output_dim = input_dim 
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(), # Using LeakyReLU as approximation for Swish if not available
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.output_dim)
            # Final activation is omitted here and will be determined by the loss function (US-2.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the state vector.
        :param x: State tensor (batch_size, input_dim)
        :return: Action tensor (batch_size, output_dim)
        """
        return self.net(x)

    def get_output_shape(self):
        """Helper to return the expected output shape for compatibility checks."""
        return self.output_dim
