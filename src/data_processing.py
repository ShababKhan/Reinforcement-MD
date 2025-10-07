"""
This module provides functions for loading molecular dynamics trajectories,
extracting and processing atomic coordinates, and superposing structures.
"""

import MDAnalysis as mda
import numpy as np

def load_trajectory(topology_path: str, trajectory_path: str) -> mda.Universe:
    """
    Loads a molecular dynamics trajectory from specified topology and trajectory files.

    Args:
        topology_path (str): Path to the topology file (e.g., PDB, GRO).
        trajectory_path (str): Path to the trajectory file (e.g., DCD, XTC, TRR).

    Returns:
        MDAnalysis.Universe: An MDAnalysis Universe object representing the system.

    Raises:
        IOError: If the specified files cannot be found or read.
        ValueError: If MDAnalysis fails to parse the files.
    """
    try:
        universe = mda.Universe(topology_path, trajectory_path)
        return universe
    except Exception as e:
        raise IOError(f"Failed to load trajectory files: {e}") from e

def extract_crbn_heavy_atom_coordinates(universe: mda.Universe) -> np.ndarray:
    """
    Extracts the Cartesian coordinates of all heavy atoms belonging to CRBN
    from the current frame of an MDAnalysis Universe object.

    The selection targets protein atoms, excluding water and hydrogen atoms.
    Note: This function assumes the 'protein' selection correctly identifies CRBN
    within the universe. If multiple proteins are present, a more specific
    selection (e.g., by segid or resid range) might be necessary.

    Args:
        universe (MDAnalysis.Universe): An MDAnalysis Universe object, positioned
                                        at the desired frame.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (N_heavy_atoms, 3)
                       containing the [x, y, z] coordinates of CRBN's heavy atoms.
    """
    # Select heavy atoms from protein, excluding water and explicit hydrogen atoms.
    # This selection is robust as specified in the task.
    # For a real CRBN complex, one might need a more specific selection like
    # "protein and segid A and not name H*" if CRBN is in chain A.
    heavy_atoms = universe.select_atoms('protein and not resname HOH and not name H*')
    return heavy_atoms.positions

def flatten_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    Flattens a 2D array of Cartesian coordinates (N_atoms, 3) into a 1D vector.

    Args:
        coordinates (numpy.ndarray): A 2D NumPy array of shape (N, 3) where N
                                     is the number of atoms.

    Returns:
        numpy.ndarray: A 1D NumPy array of shape (N*3,) containing the flattened
                       coordinates.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Input coordinates must be a 2D array with shape (N, 3).")
    return coordinates.flatten()

def superpose_trajectory_frames(
    universe: mda.Universe,
    selection_string: str = 'protein and not resname HOH and not name H*'
) -> tuple[mda.Universe, list[np.ndarray]]:
    """
    Superposes all frames in a trajectory to the reference frame (the first frame)
    based on a specified atom selection.

    The coordinates in the `universe` object are updated *in place* for each frame.
    It also returns a list of flattened heavy atom coordinates for CRBN after superposition.

    Args:
        universe (MDAnalysis.Universe): An MDAnalysis Universe object containing
                                        the trajectory.
        selection_string (str): MDAnalysis selection string for atoms to be used
                                for superposition (e.g., 'backbone', 'protein and name CA').
                                Defaults to CRBN heavy atoms.

    Returns:
        tuple[MDAnalysis.Universe, list[numpy.ndarray]]: A tuple containing:
            - MDAnalysis.Universe: The input Universe object with frames
                                   superposed (modified in place).
            - list[numpy.ndarray]: A list where each element is a 1D NumPy array
                                   of flattened CRBN heavy atom coordinates for each frame
                                   after superposition.
    """
    if len(universe.trajectory) == 0:
        return universe, []

    # Select atoms for alignment (reference and mobile)
    reference_atoms = universe.select_atoms(selection_string)
    if not reference_atoms:
        raise ValueError(f"No atoms selected for alignment with selection string: {selection_string}")

    # Set the first frame as the reference
    universe.trajectory[0]
    ref_coordinates = reference_atoms.positions.copy() # Need a copy as universe.trajectory will move

    superposed_crbn_coords = []

    # Iterate through the trajectory and superpose each frame
    for ts in universe.trajectory:
        mobile_atoms = universe.select_atoms(selection_string)
        # Perform superposition. `MDAnalysis.transform.superpose` modifies the mobile_atoms' positions
        # and therefore the `ts` object's coordinates in place.
        mda.transform.superpose(mobile_atoms, reference_atoms, ref_pos=ref_coordinates)

        # After superposition, extract and flatten the CRBN heavy atom coordinates for this frame
        crbn_coords_frame = extract_crbn_heavy_atom_coordinates(universe)
        superposed_crbn_coords.append(flatten_coordinates(crbn_coords_frame))

    # Reset trajectory to the first frame or beginning for consistency if needed elsewhere
    universe.trajectory.rewind()
    return universe, superposed_crbn_coords
