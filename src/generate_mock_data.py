import numpy as np
import os

# Configuration based on paper details (S08, S05)
N_SAMPLES = 10000
VECTOR_LENGTH = 9696  # Corresponds to flattened Cartesian coordinates of all heavy atoms
N_CV = 3
TRAIN_SPLIT = 0.8

def generate_mock_crbn_data(n_samples: int, vector_length: int, n_cv: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates mock molecular dynamics input data (structures and collective variables)
    mimicking the rMD paper's requirements.

    The structures (Cartesian coordinates) are generated to have a specific flattened
    length (9696). The **structure superposition** requirement (S03) is approximated
    by generating centered data around a mean structure.

    @param n_samples: The total number of data frames to generate.
    @param vector_length: The required length of the flattened structure vector (9696).
    @param n_cv: The number of collective variables (3).
    @return: A tuple containing (structures, collective_variables).
    @raises ValueError: If n_samples, vector_length, or n_cv are zero or negative.
    """
    if n_samples <= 0 or vector_length <= 0 or n_cv <= 0:
        raise ValueError("Sample size, vector length, and CV count must be positive.")

    print(f"Generating {n_samples} mock samples...")

    # 1. Generate Mock Structures (Approximating flattened Cartesian coordinates)
    # Structures are generated centered around zero (approximating superposition)
    # Standard deviation chosen to represent atomic motion noise.
    structure_std = 0.1  # Angstrom noise equivalent
    structures = np.random.normal(
        loc=0.0,
        scale=structure_std,
        size=(n_samples, vector_length)
    ).astype(np.float32)

    # 2. Generate Mock Collective Variables (CVs)
    # The CV space should reflect a 3D landscape (S05). We simulate some density.
    cv_std_1 = 2.0
    cv_std_2 = 1.5
    cv_std_3 = 1.0
    
    cv1 = np.random.normal(loc=5.0, scale=cv_std_1, size=n_samples)
    cv2 = np.random.normal(loc=2.0, scale=cv_std_2, size=n_samples)
    cv3 = np.random.normal(loc=-4.0, scale=cv_std_3, size=n_samples)
    
    collective_variables = np.stack([cv1, cv2, cv3], axis=1).astype(np.float32)

    print(f"Structures shape: {structures.shape}")
    print(f"CVs shape: {collective_variables.shape}")
    
    return structures, collective_variables

def split_data(structures: np.ndarray, cvs: np.ndarray, split_ratio: float) -> tuple:
    """Splits data into training and validation sets."""
    n_total = structures.shape[0]
    n_train = int(n_total * split_ratio)
    
    # Shuffle entire dataset
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train:])
    
    print(f"Data split: Train={len(train_idx)} ({len(train_idx)/n_total:.1%}), Validation={len(val_idx)} ({len(val_idx)/n_total:.1%})")
    
    return (
        structures[train_idx], cvs[train_idx],
        structures[val_idx], cvs[val_idx]
    )

if __name__ == '__main__':
    # --- Main Execution ---
    
    # Generate raw data
    all_structures, all_cvs = generate_mock_crbn_data(N_SAMPLES, VECTOR_LENGTH, N_CV)
    
    # Split data (S08: 8000 train, 2000 validation)
    X_train, Y_train, X_val, Y_val = split_data(all_structures, all_cvs, TRAIN_SPLIT)
    
    # Save data (We'll save them as simple numpy files for demonstration)
    # In a real scenario, this script would generate the required input files.
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train)
    np.save('data/Y_train.npy', Y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/Y_val.npy', Y_val)
    
    print("\\nMock data generation complete. Files saved in 'data/' directory.")
    
    # Creating a small file to confirm listing/getting works as well
    with open("tool_check_list.tmp", "w") as f:
        f.write("T1.1 Data Check")

