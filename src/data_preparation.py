"""
Data Preparation Module for rMD
Handles loading, preprocessing, and superposition of MD trajectory data
"""

import numpy as np


def load_trajectory(trajectory_path, topology_path=None):
    """
    Load molecular dynamics trajectory data.
    
    Parameters
    ----------
    trajectory_path : str
        Path to trajectory file
    topology_path : str, optional
        Path to topology file
        
    Returns
    -------
    coordinates : np.ndarray
        Array of atomic coordinates
    """
    # TODO: Implement trajectory loading with MDAnalysis
    raise NotImplementedError("Trajectory loading not yet implemented")


def superpose_structures(coordinates, reference_idx=0):
    """
    Superpose all structures to a reference structure.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (n_frames, n_atoms, 3)
    reference_idx : int
        Index of reference structure
        
    Returns
    -------
    superposed : np.ndarray
        Superposed coordinates
    """
    # TODO: Implement structure superposition
    raise NotImplementedError("Structure superposition not yet implemented")


def flatten_coordinates(coordinates):
    """
    Flatten 3D coordinates to 1D vector.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (n_frames, n_atoms, 3)
        
    Returns
    -------
    flattened : np.ndarray
        Array of shape (n_frames, n_atoms * 3)
    """
    n_frames = coordinates.shape[0]
    return coordinates.reshape(n_frames, -1)
