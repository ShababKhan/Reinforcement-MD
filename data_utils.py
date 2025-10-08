import numpy as np
from typing import Tuple

# --- Constants based on paper (Simulated) ---
# The paper mentions CRBN only (without DDB1) resulting in a length-9696 input vector.
# Since Cartesian coordinates are (x, y, z), this implies:
# N_ATOMS = 9696 / 3 = 3232 heavy atoms in CRBN
N_ATOMS = 3232
INPUT_DIM = N_ATOMS * 3  # 9696
N_FRAMES = 10000
CV_DIM = 3


def generate_synthetic_data(n_frames: int = N_FRAMES, n_atoms: int = N_ATOMS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic MD trajectory data (Cartesian coords) and corresponding 
    Collective Variables (CVs).

    This simulates a basic conformational change by introducing correlated noise 
    in the coordinates related directly to the CVs, enforcing some physical 
    relationship that the autoencoder must learn.

    @param n_frames: The number of trajectory frames to generate (default: 10000).
    @param n_atoms: The number of heavy atoms in the system (default: 3232).
    @return: A tuple (coords_flat, cvs), where:
             - coords_flat: (n_frames, n_atoms * 3) array of flattened coordinates.
             - cvs: (n_frames, 3) array of collective variables.
    """
    np.random.seed(42)  # For reproducibility

    # 1. Generate synthetic CVs (Collective Variables)
    # CVs are typically continuous and correlated. 
    # Use three simple, slightly correlated sine waves over time (frame index).
    t = np.linspace(0, 2 * np.pi, n_frames)
    cv1 = np.sin(t) + np.random.normal(0, 0.1, n_frames)
    cv2 = np.cos(t * 0.8) + np.random.normal(0, 0.1, n_frames)
    cv3 = np.sin(t * 1.2) * 0.5 + np.random.normal(0, 0.05, n_frames)
    
    cvs = np.vstack([cv1, cv2, cv3]).T  # Shape: (n_frames, 3)

    # 2. Generate a 'base' reference structure
    # Simulate a structure where atoms are centered around their mean positions
    base_structure = np.random.uniform(-10, 10, n_atoms * 3).reshape(n_atoms, 3)

    # 3. Generate the trajectory: Base structure + CV-dependent noise
    coords = []
    
    # Scale factors to control how much the CVs influence the atom positions.
    cv_scales = np.random.uniform(0.1, 0.5, size=(n_atoms, 3, 3))
    
    for i in range(n_frames):
        current_cv = cvs[i]
        
        # Introduce a collective shift related to the current CV state
        shift_x = current_cv[0] * cv_scales[:, 0, 0]
        shift_y = current_cv[1] * cv_scales[:, 1, 1]
        shift_z = current_cv[2] * cv_scales[:, 2, 2]
        
        # Combine shifts and base structure
        frame = base_structure.copy()
        frame[:, 0] += shift_x
        frame[:, 1] += shift_y
        frame[:, 2] += shift_z

        # Add minor random thermal noise
        frame += np.random.normal(0, 0.1, size=(n_atoms, 3))
        
        coords.append(frame)
        
    coords_3d = np.array(coords) # Shape: (n_frames, n_atoms, 3)
    
    # 4. Flatten the coordinates, but first, the crucial superposition step (T1.3)
    coords_flat = superpose_and_flatten(coords_3d)

    print(f"Generated synthetic data: {n_frames} frames, {n_atoms} atoms.")
    print(f"Coordinates shape (flat): {coords_flat.shape}")
    print(f"CVs shape: {cvs.shape}")

    return coords_flat, cvs


def superpose_and_flatten(coords_3d: np.ndarray) -> np.ndarray:
    """
    Performs the crucial preprocessing step:
    1. Superposes all frames onto the first frame (eliminating global R/T).
    2. Flattens the 3D coordinates into a 1D vector per frame.
    
    The implementation simplifies the complexity of structural superposition by 
    just aligning the center of geometry (COG) of all frames to the COG of the
    reference structure (Frame 0). This is a simple analogue to eliminating 
    global translation/rotation for synthetic data.
    
    @param coords_3d: Input array of shape (N_frames, N_atoms, 3).
    @return: Superposed and flattened array of shape (N_frames, N_atoms * 3).
    """
    n_frames, n_atoms, _ = coords_3d.shape
    
    # 1. Get the reference structure (Frame 0)
    ref_coords = coords_3d[0]
    
    # 2. Calculate the center of geometry (COG) for the reference structure
    # In real MD, more complex superposition (e.g., RMSD minimization via alignment)
    # would be used, especially for rotation. We simplify by centering.
    ref_cog = np.mean(ref_coords, axis=0) # Shape (3,)
    
    # 3. Translate the reference structure so its COG is at the origin (0,0,0)
    centered_ref_coords = ref_coords - ref_cog
    
    # 4. Superpose (Center) all other frames to the reference's centering.
    superposed_coords = np.zeros_like(coords_3d)
    
    for i in range(n_frames):
        current_frame = coords_3d[i]
        
        # Get the current frame's COG
        current_cog = np.mean(current_frame, axis=0)
        
        # Center the current frame relative to its own COG
        centered_frame = current_frame - current_cog
        
        # For simplicity in synthetic data, we align based on COG translation only.
        # In real-world, a rotation matrix would also be applied here (Kabsch algorithm/SVD)
        # to minimize RMSD, but for training the NN on general topology, centering is often sufficient 
        # for a first pass, as the network learns to map internal coordinates, not space coordinates.
        superposed_coords[i] = centered_frame
        
    # 5. Flatten the resulting coordinates
    coords_flat = superposed_coords.reshape(n_frames, n_atoms * 3)
    
    return coords_flat


if __name__ == '__main__':
    # Example usage and saving the data for later consumption
    coords_flat_data, cvs_data = generate_synthetic_data()

    # Save data for use in the rmd_model and train scripts
    np.save('synthetic_coords.npy', coords_flat_data)
    np.save('synthetic_cvs.npy', cvs_data)
    print("Synthetic data saved to 'synthetic_coords.npy' and 'synthetic_cvs.npy'.")