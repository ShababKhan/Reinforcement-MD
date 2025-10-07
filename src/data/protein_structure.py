"""
Classes for representing and manipulating protein structures.
"""
import numpy as np
from scipy.spatial.transform import Rotation

class ProteinStructure:
    """
    Class for representing and manipulating protein structures.
    """
    
    def __init__(self, coordinates=None, atom_names=None, residue_names=None, 
                 chain_ids=None, residue_ids=None, element_symbols=None):
        """
        Initialize a protein structure.
        
        Parameters:
        -----------
        coordinates : numpy.ndarray, optional
            Array of atomic coordinates with shape (n_atoms, 3).
        atom_names : list, optional
            List of atom names.
        residue_names : list, optional
            List of residue names.
        chain_ids : list, optional
            List of chain identifiers.
        residue_ids : list, optional
            List of residue identifiers.
        element_symbols : list, optional
            List of element symbols.
        """
        self.coordinates = coordinates
        self.atom_names = atom_names or []
        self.residue_names = residue_names or []
        self.chain_ids = chain_ids or []
        self.residue_ids = residue_ids or []
        self.element_symbols = element_symbols or []
        
        # Dictionary mapping atom types to atomic weights
        self.atomic_weights = {
            'H': 1.008,
            'C': 12.011,
            'N': 14.007,
            'O': 15.999,
            'S': 32.06,
            'P': 30.974,
            'F': 18.998,
            'CL': 35.45,
            'BR': 79.904,
            'I': 126.904,
            'FE': 55.845,
            'ZN': 65.38,
            'MG': 24.305,
            'CA': 40.078
        }
    
    @property
    def n_atoms(self):
        """
        Get the number of atoms in the structure.
        
        Returns:
        --------
        int
            Number of atoms.
        """
        if self.coordinates is None:
            return 0
        return len(self.coordinates)
    
    def get_heavy_atom_indices(self):
        """
        Get indices of heavy atoms (non-hydrogen).
        
        Returns:
        --------
        list
            List of indices of heavy atoms.
        """
        if not self.element_symbols:
            raise ValueError("Element symbols not available")
            
        return [i for i, elem in enumerate(self.element_symbols) if elem.upper() != 'H']
    
    def get_residue_indices(self, residue_id, chain_id=None):
        """
        Get atom indices for a specific residue.
        
        Parameters:
        -----------
        residue_id : int
            Residue identifier.
        chain_id : str, optional
            Chain identifier.
            
        Returns:
        --------
        list
            List of atom indices belonging to the specified residue.
        """
        if not self.residue_ids:
            raise ValueError("Residue information not available")
            
        indices = []
        for i, res_id in enumerate(self.residue_ids):
            if res_id == residue_id:
                if chain_id is None or (self.chain_ids and self.chain_ids[i] == chain_id):
                    indices.append(i)
        
        return indices
    
    def get_center_of_mass(self, atom_indices=None):
        """
        Calculate the center of mass for a set of atoms.
        
        Parameters:
        -----------
        atom_indices : list, optional
            List of atom indices to include in the calculation.
            If None, all atoms are included.
            
        Returns:
        --------
        numpy.ndarray
            Center of mass coordinates.
        """
        if self.coordinates is None:
            raise ValueError("No coordinates available")
            
        if atom_indices is None:
            atom_indices = range(self.n_atoms)
            
        total_mass = 0.0
        weighted_sum = np.zeros(3)
        
        for idx in atom_indices:
            if idx < 0 or idx >= self.n_atoms:
                raise IndexError(f"Atom index {idx} out of bounds")
                
            # Get atom mass
            if self.element_symbols and idx < len(self.element_symbols):
                element = self.element_symbols[idx].upper()
                mass = self.atomic_weights.get(element, 12.0)  # Default to carbon mass
            else:
                mass = 12.0  # Default to carbon mass
                
            total_mass += mass
            weighted_sum += mass * self.coordinates[idx]
            
        if total_mass == 0:
            raise ValueError("Total mass is zero")
            
        return weighted_sum / total_mass
    
    def superpose(self, reference, atom_indices=None, ref_atom_indices=None):
        """
        Superpose the structure onto a reference structure using Kabsch algorithm.
        
        Parameters:
        -----------
        reference : ProteinStructure
            Reference structure.
        atom_indices : list, optional
            List of atom indices to use for superposition.
            If None, all atoms are used.
        ref_atom_indices : list, optional
            List of atom indices in the reference structure.
            If None, the same indices as atom_indices are used.
            
        Returns:
        --------
        float
            RMSD between the superposed atoms.
        """
        if self.coordinates is None:
            raise ValueError("No coordinates available")
            
        if reference.coordinates is None:
            raise ValueError("No reference coordinates available")
            
        if atom_indices is None:
            atom_indices = range(self.n_atoms)
            
        if ref_atom_indices is None:
            ref_atom_indices = atom_indices
            
        if len(atom_indices) != len(ref_atom_indices):
            raise ValueError("Number of atoms must match between structures")
            
        # Extract coordinates for superposition
        coords = self.coordinates[atom_indices]
        ref_coords = reference.coordinates[ref_atom_indices]
        
        # Calculate centroids
        centroid = np.mean(coords, axis=0)
        ref_centroid = np.mean(ref_coords, axis=0)
        
        # Center both coordinate sets
        coords_centered = coords - centroid
        ref_coords_centered = ref_coords - ref_centroid
        
        # Calculate covariance matrix
        covariance = np.dot(coords_centered.T, ref_coords_centered)
        
        # Singular value decomposition
        U, S, Vt = np.linalg.svd(covariance)
        
        # Calculate rotation matrix
        rotation_matrix = np.dot(U, Vt)
        
        # Check if we need to correct for a reflection
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = np.dot(U, Vt)
        
        # Calculate translation
        translation = ref_centroid - np.dot(centroid, rotation_matrix)
        
        # Apply rotation and translation to all coordinates
        self.coordinates = np.dot(self.coordinates, rotation_matrix) + translation
        
        # Calculate RMSD between superposed atoms
        superposed_coords = np.dot(coords, rotation_matrix) + translation
        squared_diff = np.sum((superposed_coords - ref_coords) ** 2, axis=1)
        rmsd = np.sqrt(np.mean(squared_diff))
        
        return rmsd
    
    def calculate_rmsd(self, reference, atom_indices=None, ref_atom_indices=None):
        """
        Calculate RMSD between this structure and a reference without modifying either.
        
        Parameters:
        -----------
        reference : ProteinStructure
            Reference structure.
        atom_indices : list, optional
            List of atom indices to use for RMSD calculation.
            If None, all atoms are used.
        ref_atom_indices : list, optional
            List of atom indices in the reference structure.
            If None, the same indices as atom_indices are used.
            
        Returns:
        --------
        float
            RMSD between the selected atoms.
        """
        if self.coordinates is None:
            raise ValueError("No coordinates available")
            
        if reference.coordinates is None:
            raise ValueError("No reference coordinates available")
            
        # Create a copy of this structure
        temp_structure = ProteinStructure(
            coordinates=self.coordinates.copy(),
            atom_names=self.atom_names.copy() if self.atom_names else None,
            residue_names=self.residue_names.copy() if self.residue_names else None,
            chain_ids=self.chain_ids.copy() if self.chain_ids else None,
            residue_ids=self.residue_ids.copy() if self.residue_ids else None,
            element_symbols=self.element_symbols.copy() if self.element_symbols else None
        )
        
        # Superpose the copy onto the reference
        rmsd = temp_structure.superpose(reference, atom_indices, ref_atom_indices)
        
        return rmsd
    
    def flatten_coordinates(self):
        """
        Flatten the coordinates to a 1D array.
        
        Returns:
        --------
        numpy.ndarray
            Flattened coordinates.
        """
        if self.coordinates is None:
            raise ValueError("No coordinates available")
            
        return self.coordinates.flatten()
    
    def unflatten_coordinates(self, flat_coords):
        """
        Reshape a flattened 1D array back to 3D coordinates.
        
        Parameters:
        -----------
        flat_coords : numpy.ndarray
            Flattened coordinates.
            
        Returns:
        --------
        numpy.ndarray
            Reshaped coordinates with shape (n_atoms, 3).
        """
        n_atoms = len(flat_coords) // 3
        if len(flat_coords) != n_atoms * 3:
            raise ValueError("Flat coordinates length must be a multiple of 3")
            
        return flat_coords.reshape(n_atoms, 3)
    
    def update_coordinates(self, new_coords):
        """
        Update the structure coordinates.
        
        Parameters:
        -----------
        new_coords : numpy.ndarray
            New coordinates, either in shape (n_atoms, 3) or flattened.
        """
        if len(new_coords.shape) == 1:
            # Flattened coordinates
            self.coordinates = self.unflatten_coordinates(new_coords)
        else:
            # Already in shape (n_atoms, 3)
            if new_coords.shape[0] != self.n_atoms:
                raise ValueError(f"Expected {self.n_atoms} atoms but got {new_coords.shape[0]}")
                
            if new_coords.shape[1] != 3:
                raise ValueError(f"Expected 3 coordinates per atom but got {new_coords.shape[1]}")
                
            self.coordinates = new_coords.copy()
    
    def calculate_collective_variables(self, domain_definitions):
        """
        Calculate collective variables based on domain definitions.
        
        Parameters:
        -----------
        domain_definitions : dict
            Dictionary defining domains and their atom indices.
            
        Returns:
        --------
        dict
            Dictionary of calculated collective variables.
        """
        if self.coordinates is None:
            raise ValueError("No coordinates available")
            
        cv_values = {}
        domain_centroids = {}
        
        # Calculate center of mass for each domain
        for domain, indices in domain_definitions.items():
            domain_centroids[domain] = self.get_center_of_mass(indices)
            
        # Calculate distances between domain centroids
        for i, domain1 in enumerate(domain_centroids.keys()):
            for domain2 in list(domain_centroids.keys())[i+1:]:
                distance = np.linalg.norm(domain_centroids[domain1] - domain_centroids[domain2])
                cv_values[f"distance_{domain1}_{domain2}"] = distance
                
        return cv_values