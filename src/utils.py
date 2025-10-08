"""
Utility Functions for rMD
Common helper functions for data manipulation and validation
"""

import numpy as np


def calculate_rmsd(coords1, coords2):
    """
    Calculate root mean square deviation between two coordinate sets.
    
    Parameters
    ----------
    coords1 : np.ndarray
        First coordinate set
    coords2 : np.ndarray
        Second coordinate set
        
    Returns
    -------
    rmsd : float
        RMSD value in Angstroms
    """
    diff = coords1 - coords2
    squared_diff = np.square(diff)
    msd = np.mean(squared_diff)
    rmsd = np.sqrt(msd)
    return rmsd


def split_train_val(data, val_fraction=0.2, shuffle=True, random_seed=42):
    """
    Split data into training and validation sets.
    
    Parameters
    ----------
    data : np.ndarray or tuple
        Data to split
    val_fraction : float
        Fraction of data to use for validation
    shuffle : bool
        Whether to shuffle before splitting
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    train_data : np.ndarray or tuple
        Training data
    val_data : np.ndarray or tuple
        Validation data
    """
    np.random.seed(random_seed)
    
    if isinstance(data, tuple):
        n_samples = len(data[0])
    else:
        n_samples = len(data)
    
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - val_fraction))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    if isinstance(data, tuple):
        train_data = tuple(d[train_indices] for d in data)
        val_data = tuple(d[val_indices] for d in data)
    else:
        train_data = data[train_indices]
        val_data = data[val_indices]
    
    return train_data, val_data


def save_model_checkpoint(model, filepath, metadata=None):
    """
    Save model checkpoint with metadata.
    
    Parameters
    ----------
    model : keras.Model
        Model to save
    filepath : str
        Path to save checkpoint
    metadata : dict, optional
        Additional metadata to save
    """
    # TODO: Implement model checkpointing
    raise NotImplementedError("Model checkpointing not yet implemented")


def load_model_checkpoint(filepath):
    """
    Load model checkpoint.
    
    Parameters
    ----------
    filepath : str
        Path to checkpoint
        
    Returns
    -------
    model : keras.Model
        Loaded model
    metadata : dict
        Checkpoint metadata
    """
    # TODO: Implement checkpoint loading
    raise NotImplementedError("Checkpoint loading not yet implemented")
