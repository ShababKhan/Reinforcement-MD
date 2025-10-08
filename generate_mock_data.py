
import numpy as np

def generate_mock_data():
    """
    Generates synthetic/mock input coordinate data and collective variable data.

    This function creates two NumPy arrays:
    - input_coords: Represents flattened Cartesian coordinates of protein structures.
                    Shape: (10000, 9696) with float values.
    - cv_coords: Represents 3-dimensional collective variable data.
                 Shape: (10000, 3) with float values.

    The values are generated randomly between 0 and 1 for simplicity in this
    mock data generation.

    @return: A tuple containing:
             - input_coords (np.ndarray): Synthetic input coordinates.
             - cv_coords (np.ndarray): Synthetic collective variable coordinates.
    @rtype: tuple
    """
    num_samples = 10000
    input_dim = 9696
    cv_dim = 3

    # Generate random float values for input_coords (e.g., between 0 and 1)
    input_coords = np.random.rand(num_samples, input_dim).astype(np.float32)

    # Generate random float values for cv_coords (e.g., between 0 and 1)
    cv_coords = np.random.rand(num_samples, cv_dim).astype(np.float32)

    return input_coords, cv_coords

if __name__ == "__main__":
    # Example usage when run as a script
    input_data, cv_data = generate_mock_data()
    print(f"Generated input_coords shape: {input_data.shape}")
    print(f"Generated cv_coords shape: {cv_data.shape}")
    print(f"Sample input_coords (first 5 elements of first sample): {input_data[0, :5]}")
    print(f"Sample cv_coords (first sample): {cv_data[0]}")
