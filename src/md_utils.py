import numpy as np
import json
import os
from typing import Dict, Any
from numpy.typing import NDArray

def calculate_rmsd(ref_coords: NDArray[np.float64], coords: NDArray[np.float64]) -> np.float64:
    """
    Calculates the Root Mean Square Deviation (RMSD) between two sets of coordinates.

    This function utilizes the Kabsch algorithm (via SVD) to find the optimal
    rotational and translational superposition of the 'coords' onto the
    'ref_coords' before calculating the minimal RMSD.

    Args:
        ref_coords (NDArray[np.float64]): Reference set of coordinates (N x 3 array).
        coords (NDArray[np.float64]): Current set of coordinates (N x 3 array).

    Returns:
        np.float64: The minimum RMSD value after optimal superposition.

    Raises:
        ValueError: If coordinate arrays are not N x 3 or if their shapes do not match.
        TypeError: If inputs are not NumPy arrays.

    Methodology:
        1. Checks for shape and type validity.
        2. Centers both coordinate sets by subtracting their respective centroids.
        3. Computes the covariance matrix H.
        4. Applies Singular Value Decomposition (SVD) to H.
        5. Computes the optimal rotation matrix using the Kabsch formula.
        6. Calculates the final minimal RMSD.

    Note:
        Reference for RMSD calculation: Kabsch, W. (1976). A solution of the
        optimal rotation to relate two sets of vectors. Acta Cryst. A32, 922-923.
    """
    if not isinstance(ref_coords, np.ndarray) or not isinstance(coords, np.ndarray):
        raise TypeError("Inputs must be NumPy arrays.")

    if ref_coords.shape != coords.shape or ref_coords.ndim != 2 or ref_coords.shape[1] != 3:
        raise ValueError("Coordinate arrays must be N x 3 and have matching shapes.")

    n_atoms = ref_coords.shape[0]
    
    # 1. Centroid subtraction (Translation)
    ref_centroid = ref_coords.mean(axis=0)
    current_centroid = coords.mean(axis=0)

    p = ref_coords - ref_centroid
    q = coords - current_centroid

    # 2. Covariance matrix H
    h = p.T @ q

    # 3. SVD H = U S V.T
    u, s, vh = np.linalg.svd(h)
    v = vh.T

    # 4. Determine the sign of the rotation (d)
    d = np.sign(np.linalg.det(v @ u.T))
    
    # 5. Create the correction matrix for reflection/inversion
    e = np.eye(3)
    e[2, 2] = d

    # 6. Sum of products of singular values
    sum_s = (s[0] + s[1] + s[2] * d)
    
    # 7. E0 term (Sum of squares before superposition)
    e0 = np.sum(p * p) + np.sum(q * q)
    
    # 8. Calculate minimal RMSD using the formula
    rmsd_squared = (e0 - 2 * sum_s) / n_atoms
    
    # Ensure non-negative result due to potential floating point errors
    return np.sqrt(max(0.0, rmsd_squared))

def parse_config(config_path: str) -> Dict[str, Any]:
    """
    Reads and parses a project configuration file in JSON format.

    Args:
        config_path (str): The full path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    if os.path.getsize(config_path) == 0:
        # Raise a concise error for empty file before attempting JSON decode
        raise json.JSONDecodeError("Configuration file is empty.", config_path, 0)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        # Re-raise the JSON error with context
        raise json.JSONDecodeError(f"Invalid JSON format in file: {config_path}. {e.msg}", e.doc, e.pos)

# --- Master Documentation Update ---
# Update to PROJECT_BLUEPRINT.md

__doc_update__ = """
### Sprint 1: Collective Variables & MD Simulation Setup

#### Methodology Update: Core Utilities (`src/md_utils.py`)

*   **`calculate_rmsd`**: Implements the core RMSD metric required for trajectory comparison (Figure 3 in the source paper). We use the mathematically rigorous **Kabsch algorithm** based on Singular Value Decomposition (SVD) to ensure rotationally and translationally invariant minimal RMSD calculation. This is necessary for robust comparison of molecular conformations.
*   **`parse_config`**: Handles the necessary foundational task of reading project-wide settings from a standard JSON configuration file, ensuring a clean separation of code and simulation parameters. This supports the modular design required by the project plan.
"""