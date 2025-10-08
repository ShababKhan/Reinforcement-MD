# src/data_utils.py

"""
Data Utility Module for Reinforced Molecular Dynamics (rMD) Project.

This module contains functions to abstract the parsing of MD data,
define the Collective Variable (CV) structure, and generate mock datasets
that mimic the input requirements for the informed autoencoder.
"""

import numpy as np
from typing import Tuple, Dict

# --- Constants derived from the research paper ---
# The paper states input vectors are length-9696 (all heavy atoms in CRBN)
# We will use this as the expected feature dimension for mock data.
INPUT_DIMENSION = 9696
NUM_SAMPLES = 10000
CV_DIMENSION = 3

class MDDataStructure:
    """
    A container class to represent the necessary components of a single data point
    as required by the rMD training process.
    """
    def __init__(self, structure_vector: np.ndarray, cv_vector: np.ndarray):
        """
        Initialize a single structured data point.

        @param structure_vector: Flattened Cartesian coordinates vector (length 9696).
        @param cv_vector: Collective Variables vector (length 3).
        """
        if structure_vector.shape[0] != INPUT_DIMENSION:
            raise ValueError(
                f"Structure vector must have dimension {INPUT_DIMENSION}, got {structure_vector.shape[0]}."
            )
        if cv_vector.shape[0] != CV_DIMENSION:
            raise ValueError(
                f"CV vector must have dimension {CV_DIMENSION}, got {cv_vector.shape[0]}."
            )
        self.structure_vector = structure_vector
        self.cv_vector = cv_vector

def abstract_superposition_utility(structure_data: np.ndarray) -> np.ndarray:
    """
    TASK 2.1 Placeholder: Defines the logical steps for structure superposition.
    
    In a real scenario, this function would read PDB/XYZ data, translate/rotate
    each frame to align its center of mass or a reference frame (e.g., CA backbone)
    to a central point/orientation, and return the coordinates.
    
    For mock generation, we return the input data, simulating the result of this step.
    
    @param structure_data: A raw structure representation (abstracted).
    @return: The superposition-corrected structure data.
    """
    # Simulating the result of superposition: coordinates are already zero-centered/aligned.
    return structure_data

def calculate_mock_cvs() -> np.ndarray:
    """
    TASK 2.2 Placeholder: Logic to define and generate mock CV coordinates
    that span a 3D space suitable for mapping (mimicking the FE landscape).
    
    We generate random points within a cube to simulate the input space (CVs)
    which will serve as the target for Loss 1.
    
    @return: A mock 3D CV vector.
    """
    # Simulating CV space generation, akin to points on a grid or calculated path.
    # We use a simple Gaussian distribution for mock data coherence.
    mock_cv = np.random.normal(loc=0.0, scale=1.0, size=CV_DIMENSION).astype(np.float32)
    return mock_cv

def generate_r_md_mock_dataset(
    num_samples: int = NUM_SAMPLES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TASK 2.3: Generates a mock dataset mimicking the rMD simulation input.

    The dataset simulates 10,000 trajectory frames, where each frame is
    represented by a 9696-dimensional vector (flattened heavy-atom coordinates
    after superposition), alongside its corresponding 3D CV target.
    This serves as the input for initializing the autoencoder training.

    @param num_samples: The total number of mock trajectory frames to generate.
    @return: A tuple containing (mock_structure_vectors, mock_cv_targets).
    @raises ValueError: If the generated data dimensions do not match expectations.
    """
    print(f"Generating {num_samples} mock data points...")
    
    # 1. Generate Mock Structure Vectors (Input for Encoder/Loss 2)
    # In reality, these would come from MD/PDB data, but here they are random floats.
    X_structures = np.random.uniform(
        low=-0.5, high=0.5, size=(num_samples, INPUT_DIMENSION)
    ).astype(np.float32)
    
    # 2. Generate Mock CV Targets (Target for LatentLoss/Loss 1)
    # This simulates the pre-calculated Free Energy Map coordinates.
    # We use the logic defined in calculate_mock_cvs in a vectorized way.
    Y_cvs = np.random.uniform(
        low=-5.0, high=5.0, size=(num_samples, CV_DIMENSION)
    ).astype(np.float32)
    
    # --- Validation Check ---
    if X_structures.shape != (num_samples, INPUT_DIMENSION):
        raise ValueError(
            f"Mock structure generation failed. Expected shape {(num_samples, INPUT_DIMENSION)}, got {X_structures.shape}"
        )
    if Y_cvs.shape != (num_samples, CV_DIMENSION):
        raise ValueError(
             f"Mock CV generation failed. Expected shape {(num_samples, CV_DIMENSION)}, got {Y_cvs.shape}"
        )
        
    print(f"Mock dataset successfully generated. Structure shape: {X_structures.shape}, CV shape: {Y_cvs.shape}")
    
    return X_structures, Y_cvs

if __name__ == '__main__':
    # Simple test run when module is executed directly
    print("Running internal test of data_utils module...")
    
    # Test CV calculation placeholder
    test_cv = calculate_mock_cvs()
    print(f"Single mock CV test: {test_cv}, Shape: {test_cv.shape}")

    # Test full dataset generation
    # To avoid dependency on OS file writing (which requires creating dirs), we skip the save part for this test block.
    X, Y = generate_r_md_mock_dataset(num_samples=500) 
    
    # Test structure object initialization
    try:
        data_point = MDDataStructure(X[0], Y[0])
        print(f"MDDataStructure successfully initialized for first element.")
    except ValueError as e:
        print(f"Error during MDDataStructure test: {e}")

    print("Data utility module tests complete.")