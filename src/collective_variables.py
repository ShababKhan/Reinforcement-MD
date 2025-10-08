"""
Collective Variables Module for rMD
Implements CV calculations for CRBN conformational analysis
"""

import numpy as np


def compute_center_of_mass(coordinates, masses=None):
    """
    Compute center of mass for a set of atoms.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Atomic coordinates of shape (n_atoms, 3)
    masses : np.ndarray, optional
        Atomic masses of shape (n_atoms,)
        
    Returns
    -------
    com : np.ndarray
        Center of mass coordinates (3,)
    """
    if masses is None:
        # Use uniform masses if not provided
        return np.mean(coordinates, axis=0)
    else:
        total_mass = np.sum(masses)
        weighted_coords = coordinates * masses[:, np.newaxis]
        return np.sum(weighted_coords, axis=0) / total_mass


def calculate_cv_coordinates(structure, domain_definitions):
    """
    Calculate collective variable coordinates (CV1, CV2, CV3) for CRBN.
    
    As described in Figure S1 of the paper:
    - CV1, CV2, CV3 are the three edge distances of a tetrahedron
    - Vertices are COMs of CRBN-NTD, CRBN-CTD, CRBN-HBD, DDB1-BPC domains
    
    Parameters
    ----------
    structure : np.ndarray
        Atomic coordinates
    domain_definitions : dict
        Dictionary mapping domain names to atom indices
        
    Returns
    -------
    cv_coords : np.ndarray
        Array of shape (3,) containing [CV1, CV2, CV3]
    """
    # TODO: Implement CV calculation based on domain COMs
    raise NotImplementedError("CV calculation not yet implemented")


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Parameters
    ----------
    point1 : np.ndarray
        Coordinates of first point
    point2 : np.ndarray
        Coordinates of second point
        
    Returns
    -------
    distance : float
        Euclidean distance
    """
    return np.linalg.norm(point1 - point2)
