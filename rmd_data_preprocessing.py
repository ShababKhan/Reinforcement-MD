import numpy as np
from scipy.spatial.transform import Rotation


def flatten_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Flattens a 3D array of Cartesian coordinates into a 1D array.

    The input coordinates are typically of shape (num_atoms, 3), where each row
    represents the (x, y, z) coordinates of an atom. The function
    transforms this into a single 1D array, which is often required
    as input for neural networks.

    @param coords: A NumPy array of shape (num_atoms, 3) representing
                   the Cartesian coordinates of atoms.
    @return: A 1D NumPy array of shape (num_atoms * 3,) containing the
             flattened coordinates.
    @raises TypeError: If `coords` is not a NumPy array.
    @raises ValueError: If `coords` does not have a shape compatible with (N, 3).
    """
    if not isinstance(coords, np.ndarray):
        raise TypeError("Input 'coords' must be a NumPy array.")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Input 'coords' must be a 2D array with shape (N, 3).")

    return coords.reshape(-1)


def superpose_structures(
    structures: np.ndarray, reference_structure: np.ndarray
) -> np.ndarray:
    """
    Superposes a set of protein structures to a reference structure,
    eliminating global rotational and translational degrees of freedom.

    This function applies the Kabsch algorithm (or similar) to each structure
    in the input set, aligning it optimally with the reference structure
    by minimizing the Root Mean Square Deviation (RMSD).

    @param structures: A NumPy array of protein structures. Expected shape is
                       (num_frames, num_atoms, 3).
    @param reference_structure: A single NumPy array representing the
                                reference structure. Expected shape is
                                (num_atoms, 3).
    @return: A NumPy array of superposed structures, with the same shape as
             the input 'structures' (num_frames, num_atoms, 3).
    @raises TypeError: If inputs are not NumPy arrays.
    @raises ValueError: If shapes are incompatible (e.g., differing number of atoms,
                        incorrect dimensions).
    """
    if not isinstance(structures, np.ndarray) or not isinstance(
        reference_structure, np.ndarray
    ):
        raise TypeError("Inputs 'structures' and 'reference_structure' must be NumPy arrays.")

    if reference_structure.ndim != 2 or reference_structure.shape[1] != 3:
        raise ValueError(
            "Reference structure must be a 2D array with shape (num_atoms, 3)."
        )
    if structures.ndim != 3 or structures.shape[2] != 3:
        raise ValueError(
            "Input 'structures' must be a 3D array with shape (num_frames, num_atoms, 3)."
        )

    if structures.shape[1] != reference_structure.shape[0]:
        raise ValueError(
            "Number of atoms in 'structures' must match 'reference_structure'."
        )

    num_frames, _, _ = structures.shape
    superposed_structures = np.zeros_like(structures)

    # Calculate centroid of the reference structure
    ref_centroid = np.mean(reference_structure, axis=0)
    ref_centered = reference_structure - ref_centroid

    for i in range(num_frames):
        current_structure = structures[i]

        # Calculate centroid of the current structure
        current_centroid = np.mean(current_structure, axis=0)
        current_centered = current_structure - current_centroid

        # Find optimal rotation using Kabsch algorithm (via scipy's align_vectors)
        try:
            # Rotation.align_vectors(a, b) finds the rotation R such that a @ R approximates b
            rotation_obj, _ = Rotation.align_vectors(current_centered, ref_centered)
            rotation_matrix = rotation_obj.as_matrix()
        except ValueError as e:
            # In rare cases, align_vectors might fail (e.g., highly degenerate points).
            # Fallback to applying only translation.
            print(f"Warning: Alignment failed for structure {i}. Error: {e}. "
                  f"Applying only translational alignment.")
            rotation_matrix = np.eye(3) # Identity matrix for no rotation

        # Apply rotation to the current (centered) structure
        rotated_structure = current_centered @ rotation_matrix

        # Translate the rotated structure back to the reference's centroid
        superposed_structures[i] = rotated_structure + ref_centroid

    return superposed_structures
