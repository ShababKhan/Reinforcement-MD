"""Unit tests for the feature_engineering module."""
import unittest
import numpy as np
from Bio.PDB import PDBParser
from pathlib import Path
import tempfile
import os
import sys

# Attempt to configure imports for both running locally and via script
try:
    from src.feature_engineering import calculate_dihedral_angles, encode_dihedral_features
    from src.data_loader import load_pdb_structure
except ImportError:
    # Adjust path if running outside the correct env setup
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir.parent / "src"))
    from feature_engineering import calculate_dihedral_angles, encode_dihedral_features
    from data_loader import load_pdb_structure # From US 0.2

# Minimal PDB content representing a two-residue peptide (ALA-GLY)
TWO_RESIDUE_PDB = (
    # Residue 1 (ALA)
    "ATOM      1  N   ALA A   1   -4.521 -10.871  -0.540  1.00 20.00           N  \n"
    "ATOM      2  CA  ALA A   1   -4.502 -10.887   0.941  1.00 20.00           C  \n"
    "ATOM      3  C   ALA A   1   -3.411 -11.900   1.464  1.00 20.00           C  \n"
    # Residue 2 (GLY)
    "ATOM      4  N   GLY A   2   -2.274 -11.233   1.614  1.00 20.00           N  \n"
    "ATOM      5  CA  GLY A   2   -1.121 -11.942   2.133  1.00 20.00           C  \n"
    "ATOM      6  C   GLY A   2   -0.088 -10.912   1.751  1.00 20.00           C  \n"
    "TER       7                                                                  \n"
    "END                                                                          \n"
)

# Benchmarks for US 1.1 (Dihedral calculation)
EXPECTED_PHI = -1.3320092305370355
EXPECTED_PSI = 2.455209701768826
EXPECTED_OMEGA = 3.141592653589793 # $\pi$ (180 degrees)

# Benchmarks for US 1.2 (Sinusoidal Encoding)
VERIFIABLE_INPUT_ANGLES = {
    'phi': [0.0, None],          
    'psi': [np.pi / 2, None],    
    'omega': [np.pi, None]       
}
EXPECTED_ENCODED_VECTOR = np.array([1.0, 0.0, 0.0, 1.0, -1.0, 0.0]) # cos(0), sin(0), cos(pi/2), sin(pi/2), cos(pi), sin(pi)

class TestFeatureEngineering(unittest.TestCase):
    """
    Tests the dihedral angle calculation function (US 1.1) and
    the sinusoidal encoding function (US 1.2).
    """
    @classmethod
    def setUpClass(cls):
        """Setup a temporary PDB file with known geometry."""
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.temp_pdb_path = Path(cls.test_dir.name) / "dihedral_test.pdb"
        
        with open(cls.temp_pdb_path, 'w') as f:
            f.write(TWO_RESIDUE_PDB)
            
        parser = PDBParser(QUIET=True)
        cls.structure = parser.get_structure("test_dihedral", cls.temp_pdb_path)

    @classmethod
    def tearDownClass(cls):
        """Cleanup the temporary directory."""
        cls.test_dir.cleanup()

    # --- US 1.1 Tests ---
    def test_dihedral_angle_calculation(self):
        """Test that the calculated Phi, Psi, and Omega angles match known geometric values."""
        
        calculated_angles = calculate_dihedral_angles(self.structure)

        # Check list length (2 residues -> 2 elements in each list, including None padding)
        self.assertEqual(len(calculated_angles['phi']), 2)
        self.assertEqual(len(calculated_angles['psi']), 2)
        self.assertEqual(len(calculated_angles['omega']), 2)
        
        # Check Phi (GLY 2)
        self.assertAlmostEqual(calculated_angles['phi'][1], EXPECTED_PHI, places=6)

        # Check Psi (ALA 1)
        self.assertAlmostEqual(calculated_angles['psi'][0], EXPECTED_PSI, places=6)

        # Check Omega (Bond 1->2)
        self.assertAlmostEqual(calculated_angles['omega'][1], EXPECTED_OMEGA, places=6)
        
        # Check padding (terminal Nones)
        self.assertIsNone(calculated_angles['phi'][0])
        self.assertIsNone(calculated_angles['psi'][1])
        self.assertIsNone(calculated_angles['omega'][0])


    # --- US 1.2 Tests ---
    def test_sinusoidal_encoding(self):
        """Test that the sin/cos transformation is correctly applied to known angles."""
        
        # Test with manually verifiable input angles
        encoded_array = encode_dihedral_features(VERIFIABLE_INPUT_ANGLES)
        
        # Check output shape: Only ONE residue should have complete data (6 features)
        self.assertEqual(encoded_array.shape, (1, 6))
        
        # Scientific check: Check content accuracy against the known benchmark vector
        actual_vector = encoded_array[0]
        np.testing.assert_array_almost_equal(actual_vector, EXPECTED_ENCODED_VECTOR, decimal=6)

    def test_encoding_skips_none(self):
        """Test that residues containing None (terminal/missing atoms) are skipped."""

        # Mock angle output from US 1.1's structure: no index has all three defined.
        mock_angles = {
            'phi': [None, 0.5],
            'psi': [1.5, None],
            'omega': [None, 3.0]
        }
        
        encoded_array = encode_dihedral_features(mock_angles)
        
        # Expected result is an empty array since no residue has all three angles defined.
        self.assertEqual(encoded_array.shape, (0, 6))

class TestDataLoaderValidation(unittest.TestCase):
    """
    Tests the functional and scientific output of the PDB loading process (US 0.2).
    This test serves as the scientific validation script required for US 0.2.
    """

    # A minimal PDB file structure for scientific validation
    MINIMAL_PDB_CONTENT = (
        "HEADER    STRUCTURAL MODEL\n"
        "ATOM      1  N   ALA A   1      20.000  15.000  10.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      21.000  15.500  10.000  1.00 20.00           C\n"
        "TER       3      ALA A   1\n"
        "END\n"
    )

    # Benchmark Data (Expected Scientific Outcome)
    BENCHMARK_EXPECTED_COORDINATES = {
        'N': np.array([20.000, 15.000, 10.000]),
        'CA': np.array([21.000, 15.500, 10.000])
    }
    PRECISION = 3 
    
    @classmethod
    def setUpClass(cls):
        """Setup a temporary PDB file."""
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.temp_pdb_path = Path(cls.test_dir.name) / "validation_structure_02.pdb"
        with open(cls.temp_pdb_path, 'w') as f:
            f.write(cls.MINIMAL_PDB_CONTENT)
            
    @classmethod
    def tearDownClass(cls):
        """Cleanup the temporary directory."""
        cls.test_dir.cleanup()

    def test_scientific_coordinate_accuracy(self):
        """
        Validation Check: Verifies that the loaded atomic coordinates match 
        the PDB file data up to 3 decimal places (PDB standard).
        """
        structure = load_pdb_structure(str(self.temp_pdb_path))
        
        # Navigate to the first atom
        residue = list(list(list(structure)[0])[0])[0] 

        actual_coordinates = {}
        for atom in residue.get_atoms():
            actual_coordinates[atom.get_name().strip()] = atom.get_coord()

        # Check N atom coordinates
        expected_n = self.BENCHMARK_EXPECTED_COORDINATES['N']
        actual_n = actual_coordinates['N']
        np.testing.assert_array_almost_equal(actual_n, expected_n, decimal=self.PRECISION)

        # Check CA atom coordinates
        expected_ca = self.BENCHMARK_EXPECTED_COORDINATES['CA']
        actual_ca = actual_coordinates['CA']
        np.testing.assert_array_almost_equal(actual_ca, expected_ca, decimal=self.PRECISION)
