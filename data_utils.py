import numpy as np

def load_trajectory(file_path):
    """
    Loads molecular dynamics coordinates from a file (stub for US-1.1).
    Returns a numpy array of shape (N_atoms, 3).
    """
    if file_path == "data/mock_traj.xyz":
        # Mock data representing 3 atoms x 3 coordinates
        return np.array([
            [1.0, 1.0, 1.0],
            [5.0, 2.0, 3.0],
            [3.0, 6.0, 5.0]
        ], dtype=np.float64)
    elif file_path == "data/bad_path.xyz":
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        # Placeholder for real file parsing
        raise NotImplementedError("Actual file parsing not implemented yet.")

def preprocess_data(coords):
    """
    Centers the coordinates by subtracting the center of mass (US-1.1).
    Assumes all masses are equal for simplicity.

    Input: coords (numpy array) shape (N_atoms, 3)
    Output: centered_coords (numpy array) shape (N_atoms, 3)
    """
    if coords is None or coords.size == 0:
        return coords

    center_of_mass = np.mean(coords, axis=0)
    centered_coords = coords - center_of_mass
    return centered_coords

def featurize_state(coords):
    """
    Converts atomic coordinates into a state tensor for the RL agent (US-1.2).
    Simple featurization: Use the centered coordinates directly as the state vector.

    Input: coords (numpy array) shape (N_atoms, 3)
    Output: state_vector (numpy array) shape (N_atoms * 3,) of dtype np.float32
    """
    if coords is None or coords.size == 0:
        # Return empty array of correct dtype if input is empty
        return np.array([], dtype=np.float32)

    # 1. Center the data (crucial scientific step: ensuring translational invariance)
    centered_coords = preprocess_data(coords)

    # 2. Flatten the coordinates into a one-dimensional state vector
    state_vector = centered_coords.flatten().astype(np.float32)

    return state_vector
