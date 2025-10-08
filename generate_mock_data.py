
import numpy as np

def generate_mock_data(num_samples=10000, input_dim=9696, cv_dim=3):
    """
    Generates synthetic/mock input_coords and cv_coords for training and validation.

    This function creates NumPy arrays that simulate protein trajectory data (input_coords)
    and corresponding collective variable data (cv_coords). The data is randomly generated
    within a reasonable range to mimic the characteristics of real molecular dynamics data
    without requiring actual simulations.

    @param num_samples: The number of data samples (e.g., trajectory frames) to generate.
                        Defaults to 10000 as per the paper.
    @param input_dim: The dimensionality of the input coordinates (flattened Cartesian
                      coordinates of heavy atoms). Defaults to 9696 for CRBN.
    @param cv_dim: The dimensionality of the collective variables. Defaults to 3 for CRBN.

    @return: A tuple containing:
             - input_coords (np.ndarray): A 2D NumPy array of shape (num_samples, input_dim)
                                          representing flattened Cartesian coordinates.
             - cv_coords (np.ndarray): A 2D NumPy array of shape (num_samples, cv_dim)
                                       representing collective variable coordinates.
    """
    print(f"Generating mock data: {num_samples} samples, input_dim={input_dim}, cv_dim={cv_dim}")

    # Generate input_coords: Simulate Cartesian coordinates.
    # Values are typically in Angstroms, so a range around +/- a few hundreds is reasonable for a large protein.
    # np.random.rand generates values in [0, 1), so scaling and shifting is applied.
    input_coords = np.random.rand(num_samples, input_dim) * 200 - 100  # Values between -100 and 100

    # Generate cv_coords: Simulate collective variables.
    # These are often distances or angles, so positive values are typical.
    # Distances in Angstroms, maybe from 0 to 50 for CVs.
    cv_coords = np.random.rand(num_samples, cv_dim) * 50

    print("Mock data generation complete.")
    return input_coords.astype(np.float32), cv_coords.astype(np.float32)

if __name__ == "__main__":
    # Example usage and shape verification
    mock_input, mock_cv = generate_mock_data()
    print(f"Shape of generated input_coords: {mock_input.shape}")
    print(f"Shape of generated cv_coords: {mock_cv.shape}")

    # Assertions for acceptance criteria
    assert mock_input.shape == (10000, 9696), "Input coordinates shape mismatch!"
    assert mock_cv.shape == (10000, 3), "CV coordinates shape mismatch!"
    print("Mock data shapes verified successfully.")
