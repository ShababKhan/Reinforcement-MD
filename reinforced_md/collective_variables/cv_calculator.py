"""
Collective variable calculator for reinforced molecular dynamics.
"""

import numpy as np
import mdtraj as md
from typing import List, Dict, Tuple, Optional
from ..data.protein_structure import ProteinStructure


class CollectiveVariableCalculator:
    """
    Calculate collective variables for the rMD approach.
    
    This class implements the collective variables (CVs) described in the paper,
    specifically the distances between centers of mass of protein domains.
    
    @param protein: ProteinStructure object
    """
    
    def __init__(self, protein: ProteinStructure):
        """
        Initialize the CV calculator with a protein structure.
        
        @param protein: ProteinStructure object
        """
        self.protein = protein
        self.domain_selections = {}
        self.domain_coms = {}
        
    def define_domain(self, domain_name: str, selection_string: str) -> None:
        """
        Define a protein domain for CV calculation.
        
        @param domain_name: Name of the domain (e.g., 'CRBN-NTD')
        @param selection_string: MDTraj selection syntax for the domain
        @return: None
        """
        self.domain_selections[domain_name] = selection_string
        
    def compute_domain_coms(self) -> Dict[str, np.ndarray]:
        """
        Compute center of mass for each defined domain across all frames.
        
        @return: Dictionary mapping domain names to COM coordinates arrays
        """
        for domain_name, selection in self.domain_selections.items():
            self.domain_coms[domain_name] = self.protein.extract_domain_com(selection)
        return self.domain_coms
        
    def compute_crbn_cvs(self) -> np.ndarray:
        """
        Compute the CRBN-specific CVs as described in the paper.
        
        The CVs are distances between the CRBN-CTD COM and other domain COMs,
        specifically CV1, CV2, and CV3 forming the tetrahedron sides.
        
        @return: Array of shape (n_frames, 3) with CV values
        @raises: ValueError if required domains are not defined
        """
        required_domains = ['CRBN-NTD', 'CRBN-CTD', 'CRBN-HBD', 'DDB1-BPC']
        for domain in required_domains:
            if domain not in self.domain_coms:
                if domain not in self.domain_selections:
                    raise ValueError(f"Domain '{domain}' not defined. Please define all required domains.")
                self.compute_domain_coms()
                break
        
        n_frames = self.domain_coms['CRBN-CTD'].shape[0]
        cvs = np.zeros((n_frames, 3))
        
        # CV1: Distance between CRBN-CTD and CRBN-NTD
        cvs[:, 0] = np.linalg.norm(
            self.domain_coms['CRBN-CTD'] - self.domain_coms['CRBN-NTD'], 
            axis=1
        )
        
        # CV2: Distance between CRBN-CTD and CRBN-HBD
        cvs[:, 1] = np.linalg.norm(
            self.domain_coms['CRBN-CTD'] - self.domain_coms['CRBN-HBD'], 
            axis=1
        )
        
        # CV3: Distance between CRBN-CTD and DDB1-BPC
        cvs[:, 2] = np.linalg.norm(
            self.domain_coms['CRBN-CTD'] - self.domain_coms['DDB1-BPC'], 
            axis=1
        )
        
        return cvs