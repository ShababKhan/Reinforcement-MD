"""
data_utils.py

Implements utility functions for preparing MD trajectory data into the format required for 
the rMD autoencoder training, including superposition, coordinate extraction, and data splitting.

This module *mocks* the actual MD/meta-eABF simulation output (P08, P13, P15) by generating 
synthetic data that matches the expected dimensions and statistical properties described in 
the paper (P16, P17).
"""
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import os

# --- Constants derived from the paper ---
INPUT_DIMENSION = 9696  # 3D coordinates for all heavy atoms in CRBN (P16)
TOTAL_FRAMES = 10000    # P17
TRAIN_RATIO = 0.8       # P19

def mock_generate_heavy_atom_data(num_frames: int, input_dim: int) -> np.ndarray:
    """
    Mocks the generation of heavy atom coordinate data from an MD trajectory.
    In a real scenario, this data would be read from PLUMED/OpenMM output files.

    @param num_frames: The total number of frames to generate (e.g., 10000).
    @param input_dim: The flattened dimension of the structure vector (9696).
    @return: A numpy array of shape (num_frames, input_dim) containing mock structural data.
    """
    print(f"Mock generating {num_frames} coordinate sets of dimension {input_dim}.")
    
    # Mock a distribution somewhat centered around a typical PDB structure in Cartesian space
    # The actual data is centered around the reference frame after superposition (P18)
    mean_structure = np.random.rand(input_dim) * 10.0 - 5.0
    std_dev = 0.5 # Small fluctuations around the mean
    
    # Generate data with some local correlations (imperfect random noise)
    data = np.random.normal(loc=mean_structure, scale=std_dev, size=(num_frames, input_dim))
    
    # Normalize data vaguely to simulate structural centering, though P18 superposition is key
    data = data - np.mean(data, axis=0)
    
    print("Mock data generation complete.")
    return data

def mock_generate_cv_data(num_frames: int, cv_dim: int = 3) -> np.ndarray:
    """
    Mocks the generation of matching 3D Collective Variable data.
    In a real implementation, this would be calculated from the same trajectory frames.
    This data is used for Loss 1 during rMD training (P23).
    
    @param num_frames: Number of frames.
    @param cv_dim: Dimension of the CV space (3 for this project).
    @return: A numpy array of shape (num_frames, cv_dim).
    """
    print(f"Mock generating {num_frames} CV vectors of dimension {cv_dim}.")
    
    # Mock CVs that might occupy a volume in the 3D space, similar to where the FE map is defined
    # Create mock points that vaguely fill a sphere/volume
    mock_cvs = np.random.uniform(low=-5.0, high=5.0, size=(num_frames, cv_dim))
    
    print("Mock CV data generation complete.")
    return mock_cvs

def prepare_rmd_dataset(input_path: str = './data') -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Handles data preparation: loads/generates coordinates, superposes (mocks), 
    splits into training and validation sets, and generates matching CV data.

    @param input_path: Directory to save/load mock data (for persistence/testing).
    @return: ((X_train, Y_train), (X_val, Y_val)) where Y is the matching CV data.
    """
    data_file = os.path.join(input_path, 'rmd_structure_data.npy')
    cv_file = os.path.join(input_path, 'rmd_cv_data.npy')
    
    os.makedirs(input_path, exist_ok=True)

    if os.path.exists(data_file) and os.path.exists(cv_file):
        print("Loading existing mock data.")
        X = np.load(data_file)
        Y = np.load(cv_file)
    else:
        # 1. Generate/Acquire Structural Data (P17)
        X = mock_generate_heavy_atom_data(TOTAL_FRAMES, INPUT_DIMENSION)
        
        # 2. Generate/Acquire CV Data (P05, P14)
        Y = mock_generate_cv_data(TOTAL_FRAMES, cv_dim=3)
        
        # 3. Superposition (Mocked - P18)
        # In reality, all structures would be aligned to the first frame's backbone.
        # Since we generated synthetic data, we rely on the noise being centered.
        print("P18: Structure superposition to first frame is mocked/implicit in generation.")
        
        np.save(data_file, X)
        np.save(cv_file, Y)

    # 4. Split Data (P19)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, train_size=TRAIN_RATIO, random_state=42
    )

    print(f"Data prepared: Train size={len(X_train)}, Validation size={len(X_val)}")
    return (X_train, Y_train), (X_val, Y_val)

if __name__ == '__main__':
    # Example usage to verify functionality (T0.1, T0.3.2 verification)
    (X_train, Y_train), (X_val, Y_val) = prepare_rmd_dataset()
    
    print("\n--- Verification ---")
    print(f"X_train shape: {X_train.shape} (Expected: ~8000, 9696)")
    print(f"Y_train shape: {Y_train.shape} (Expected: ~8000, 3)")
    print(f"X_val shape: {X_val.shape} (Expected: ~2000, 9696)")
    print(f"Y_val shape: {Y_val.shape} (Expected: ~2000, 3)")
    
    # Mock T0.2.4/T0.3.1 check
    assert X_train.shape[1] == INPUT_DIMENSION
    assert X_train.shape[0] + X_val.shape[0] == TOTAL_FRAMES
    assert Y_train.shape[1] == 3
    
    print("Sprint 0 Data Utility Check: PASSED.")