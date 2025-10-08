# File: src/feature_engineering.py 
"""
Module for calculating physics-informed features (Collective Variables) 
from the molecular structure, intended as input for the Autoencoder.
"""
from typing import List, Dict
import numpy as np
import torch
from Bio.PDB.Structure import Structure
from Bio.PDB import calc_dihedral, Atom, Residue

# Utility function to safely get a backbone atom from a residue
def _get_backbone_atoms(residue: Residue) -> Dict[str, Atom]:
    """Retrieves the N, C-alpha (CA), and C atoms for a residue."""
    
    atoms = {}
    
    # Check that it is a standard amino acid residue (or similar)
    if residue.has_id('N'):
        atoms['N'] = residue['N']
    if residue.has_id('CA'):
        atoms['CA'] = residue['CA']
    if residue.has_id('C'):
        atoms['C'] = residue['C']
        
    return atoms

def calculate_dihedral_angles(structure: Structure) -> Dict[str, List[float]]:
    """
    Calculates the backbone dihedral angles (phi, psi) for every residue 
    in the first model and chain of the structure.
    
    The Phi ($\phi$) angle is defined by atoms: C(i-1) - N(i) - CA(i) - C(i).
    The Psi ($\psi$) angle is defined by atoms: N(i) - CA(i) - C(i) - N(i+1).

    Args:
        structure (Bio.PDB.Structure.Structure): The loaded molecular structure.

    Returns:
        Dict[str, List[float]]: A dictionary containing lists of calculated 
                                angles, keyed as 'phi', 'psi', and 'omega'.
                                Values are in radians.
    """
    angles = {'phi': [], 'psi': [], 'omega': []}
    
    # Assumption: Only process the first model and first chain, common for MD setup.
    try:
        chain = list(list(structure)[0])[0] # Model 0 -> Chain 0
    except IndexError:
        return angles

    residues = [r for r in chain if r.has_id('CA')]
    
    if len(residues) < 2:
        return angles

    # Loop over residues to calculate Phi and Psi
    for i in range(len(residues)):
        prev_res = residues[i-1] if i > 0 else None
        curr_res = residues[i]
        next_res = residues[i+1] if i < len(residues) - 1 else None
        
        c_atoms = _get_backbone_atoms(curr_res)
        
        # --- Phi ($\phi$) Angle: Defined by Res i-1 and Res i ---
        phi = None
        if prev_res:
            p_atoms = _get_backbone_atoms(prev_res)
            if all(k in p_atoms for k in ['C']) and all(k in c_atoms for k in ['N', 'CA', 'C']):
                try:
                    phi = calc_dihedral(p_atoms['C'], c_atoms['N'], c_atoms['CA'], c_atoms['C'])
                except Exception:
                    pass
        angles['phi'].append(phi)
            
        # --- Psi ($\psi$) Angle: Defined by Res i and Res i+1 ---
        psi = None
        if next_res:
            n_atoms = _get_backbone_atoms(next_res)
            if all(k in c_atoms for k in ['N', 'CA', 'C']) and all(k in n_atoms for k in ['N']):
                try:
                    psi = calc_dihedral(c_atoms['N'], c_atoms['CA'], c_atoms['C'], n_atoms['N'])
                except Exception:
                    pass
        angles['psi'].append(psi)
            
        # --- Omega ($\omega$) Angle: Defined by Res i-1 and Res i ---
        omega = None
        if prev_res:
             p_atoms = _get_backbone_atoms(prev_res)
             if all(k in p_atoms for k in ['CA', 'C']) and all(k in c_atoms for k in ['N', 'CA']):
                 try:
                     omega = calc_dihedral(p_atoms['CA'], p_atoms['C'], c_atoms['N'], c_atoms['CA'])
                 except Exception:
                     pass
        # Note: Omega is technically a property of the bond, aligned roughly with the residue index
        angles['omega'].append(omega)
        
    # The current implementation handles None padding naturally in the loop.

    return angles

def encode_dihedral_features(angles: Dict[str, List[float]]) -> np.ndarray:
    """
    Transforms periodic dihedral angles into a continuous Cartesian representation 
    using sine and cosine functions.

    The transformation is $\theta \rightarrow (\cos\theta, \sin\theta)$.

    Args:
        angles (Dict[str, List[float]]): Dictionary containing lists of calculated 
                                         angles, with keys 'phi', 'psi', 'omega'. 
                                         None values are skipped.

    Returns:
        np.ndarray: A 2D NumPy array where each row represents a residue's 
                    encoded feature vector: 
                    [cos_phi, sin_phi, cos_psi, sin_psi, cos_omega, sin_omega].
                    Residues with missing data are excluded.
    """
    
    # 1. Gather all angles into lists, ensuring all required keys are present
    phi_list = angles.get('phi', [])
    psi_list = angles.get('psi', [])
    omega_list = angles.get('omega', [])
    
    # Ensure lists are of equal length (they should be per residue count)
    min_len = min(len(phi_list), len(psi_list), len(omega_list))
    
    # Truncate lists to the minimum length to ensure alignment
    phi_list = phi_list[:min_len]
    psi_list = psi_list[:min_len]
    omega_list = omega_list[:min_len]

    encoded_features = []

    # 2. Iterate through angles and perform sinusoidal encoding
    for phi, psi, omega in zip(phi_list, psi_list, omega_list):
        
        # Check for completeness (skip residues where any angle is None)
        if phi is None or psi is None or omega is None:
            continue

        # Convert to numpy array for efficient sin/cos calculation
        current_angles = np.array([phi, psi, omega])
        
        # Calculate sine and cosine components
        cos_components = np.cos(current_angles)
        sin_components = np.sin(current_angles)

        # Concatenate into the final 6-dimensional feature vector: 
        # [cos(phi), sin(phi), cos(psi), sin(psi), cos(omega), sin(omega)]
        feature_vector = np.concatenate([cos_components, sin_components])
        
        # Reshape to (1, 6) and append
        encoded_features.append(feature_vector)
    
    # 3. Stack all feature vectors into a single 2D NumPy array
    if not encoded_features:
        return np.empty((0, 6), dtype=np.float64) # Return empty array if no residues were processable
    
    return np.stack(encoded_features)
