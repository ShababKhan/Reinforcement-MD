"""
Protein structure handling for reinforced molecular dynamics.
"""

import numpy as np
import mdtraj as md
from typing import List, Tuple, Dict, Union, Optional


class ProteinStructure:
    """
    Class for handling protein structure data for reinforced molecular dynamics.
    
    This class provides functionality to load, process, and manipulate protein
    structures from PDB files or MD trajectories.
    
    @param structure_file: Path to PDB file or trajectory file
    @param topology_file: Path to topology file if loading a trajectory
    """
    
    def __init__(self, 
                 structure_file: str, 
                 topology_file: Optional[str] = None):
        """
        Initialize a protein structure object.
        
        @param structure_file: Path to PDB file or trajectory file
        @param topology_file: Path to topology file if loading a trajectory
        """
        self.structure_file = structure_file
        self.topology_file = topology_file
        self._trajectory = None
        self._load_structure()
        
    def _load_structure(self) -> None:
        """
        Load protein structure from file.
        
        @return: None
        @raises: ValueError if file cannot be loaded
        """
        try:
            if self.topology_file:
                # Load trajectory with separate topology
                self._trajectory = md.load(self.structure_file, top=self.topology_file)
            else:
                # Load PDB or trajectory with embedded topology
                self._trajectory = md.load(self.structure_file)
        except Exception as e:
            raise ValueError(f"Failed to load structure: {e}")
            
    def get_coords(self, frame_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Get Cartesian coordinates of protein atoms.
        
        @param frame_indices: Indices of frames to extract (None for all)
        @return: Array of shape (n_frames, n_atoms, 3) with coordinates in nm
        """
        if frame_indices is not None:
            traj_slice = self._trajectory[frame_indices]
            return traj_slice.xyz
        return self._trajectory.xyz
    
    def get_flattened_coords(self, frame_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Get flattened Cartesian coordinates for network input.
        
        @param frame_indices: Indices of frames to extract (None for all)
        @return: Array of shape (n_frames, n_atoms * 3) with flattened coordinates
        """
        coords = self.get_coords(frame_indices)
        return coords.reshape(coords.shape[0], -1)
    
    def align_to_reference(self, 
                          reference_frame: int = 0, 
                          atom_indices: Optional[List[int]] = None) -> None:
        """
        Align all structures to a reference frame to remove rotational and translational
        degrees of freedom, as required for the rMD approach.
        
        @param reference_frame: Index of reference frame
        @param atom_indices: Atom indices to use for alignment (None for all)
        @return: None
        """
        if atom_indices is None:
            # Use all atoms for alignment
            self._trajectory.superpose(self._trajectory[reference_frame])
        else:
            # Use specified atoms for alignment
            self._trajectory.superpose(self._trajectory[reference_frame], atom_indices=atom_indices)
            
    def select_atoms(self, selection_string: str) -> np.ndarray:
        """
        Select atoms based on MDTraj selection syntax.
        
        @param selection_string: MDTraj selection string
        @return: Array of atom indices
        """
        return self._trajectory.topology.select(selection_string)
    
    def extract_domain_com(self, selection_string: str) -> np.ndarray:
        """
        Calculate the center of mass of a domain across all frames.
        
        @param selection_string: Selection string for the domain
        @return: Array of shape (n_frames, 3) with center of mass coordinates
        """
        atom_indices = self.select_atoms(selection_string)
        domain_trajectory = self._trajectory.atom_slice(atom_indices)
        return md.compute_center_of_mass(domain_trajectory)