import unittest
import numpy as np
import MDAnalysis as mda
import os
import tempfile
from pathlib import Path

# Assume src is in the parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cv_calculations import (
    calculate_cvs,
    generate_fe_map_placeholder,
    snap_cv_to_fe_map_grid,
    CRBN_NTD_SELECTION,
    CRBN_HBD_SELECTION,
    CRBN_CTD_SELECTION,
    DDB1_BPC_SELECTION
)

class TestCVCalculations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a dummy PDB with multiple chains/residues for CV testing."""
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.test_dir.name)

        cls.dummy_complex_pdb_path = cls.temp_path / "dummy_complex.pdb"
        with open(cls.dummy_complex_pdb_path, 'w') as f:
            # CRBN Chain A: NTD (resid 1-50), HBD (resid 51-249), CTD (resid 250-320)
            # DDB1 Chain B: BPC (resid 200-600)

            # CRBN-NTD (A, 1-50) - using only resid 1, 2 for simplicity
            f.write("ATOM    1  N   ALA A   1       1.000   1.000   1.000  1.00 0.00           N  \n")
            f.write("ATOM    2  CA  ALA A   1       2.000   1.000   1.000  1.00 0.00           C  \n")
            f.write("ATOM    3  N   ALA A   2       3.000   1.000   1.000  1.00 0.00           N  \n")
            f.write("ATOM    4  CA  ALA A   2       4.000   1.000   1.000  1.00 0.00           C  \n")

            # CRBN-HBD (A, 51-249) - using only resid 51, 52
            f.write("ATOM    5  N   ALA A  51      10.000  10.000  10.000  1.00 0.00           N  \n")
            f.write("ATOM    6  CA  ALA A  51      11.000  10.000  10.000  1.00 0.00           C  \n")
            f.write("ATOM    7  N   ALA A  52      12.000  10.000  10.000  1.00 0.00           N  \n")
            f.write("ATOM    8  CA  ALA A  52      13.000  10.000  10.000  1.00 0.00           C  \n")

            # CRBN-CTD (A, 250-320) - using only resid 250, 251
            f.write("ATOM    9  N   ALA A 250      20.000  20.000  20.000  1.00 0.00           N  \n")
            f.write("ATOM   10  CA  ALA A 250      21.000  20.000  20.000  1.00 0.00           C  \n")
            f.write("ATOM   11  N   ALA A 251      22.000  20.000  20.000  1.00 0.00           N  \n")
            f.write("ATOM   12  CA  ALA A 251      23.000  20.000  20.000  1.00 0.00           C  \n")
            f.write("TER\n") # Terminator for chain A

            # DDB1-BPC (B, 200-600) - using only resid 200, 201
            f.write("ATOM   13  N   GLY B 200      30.000  30.000  30.000  1.00 0.00           N  \n")
            f.write("ATOM   14  CA  GLY B 200      31.000  30.000  30.000  1.00 0.00           C  \n")
            f.write("ATOM   15  N   GLY B 201      32.000  30.000  30.000  1.00 0.00           N  \n")
            f.write("ATOM   16  CA  GLY B 201      33.000  30.000  30.000  1.00 0.00           C  \n")
            f.write("TER\n") # Terminator for chain B
            f.write("END\n")

        cls.universe = mda.Universe(str(cls.dummy_complex_pdb_path))

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        cls.test_dir.cleanup()

    def test_calculate_cvs(self):
        cvs = calculate_cvs(self.universe)
        self.assertIsInstance(cvs, np.ndarray)
        self.assertEqual(cvs.shape, (3,))

        # Expected COMs (simple average for these dummy structures):
        # CRBN-NTD: (1.0+2.0+3.0+4.0)/4, (1.0+1.0+1.0+1.0)/4, (1.0+1.0+1.0+1.0)/4 = (2.5, 1.0, 1.0)
        # CRBN-HBD: (10.0+11.0+12.0+13.0)/4, (10.0+10.0+10.0+10.0)/4, (10.0+10.0+10.0+10.0)/4 = (11.5, 10.0, 10.0)
        # CRBN-CTD: (20.0+21.0+22.0+23.0)/4, (20.0+20.0+20.0+20.0)/4, (20.0+20.0+20.0+20.0)/4 = (21.5, 20.0, 20.0)
        # DDB1-BPC: (30.0+31.0+32.0+33.0)/4, (30.0+30.0+30.0+30.0)/4, (30.0+30.0+30.0+30.0)/4 = (31.5, 30.0, 30.0)

        com_ntd = np.array([2.5, 1.0, 1.0])
        com_hbd = np.array([11.5, 10.0, 10.0])
        com_ctd = np.array([21.5, 20.0, 20.0])
        com_bpc = np.array([31.5, 30.0, 30.0])

        expected_cv1 = np.linalg.norm(com_ctd - com_ntd) # sqrt((21.5-2.5)^2 + (20-1)^2 + (20-1)^2) = sqrt(19^2 + 19^2 + 19^2) = sqrt(3 * 19^2) = 19 * sqrt(3)
        expected_cv2 = np.linalg.norm(com_ctd - com_hbd) # sqrt((21.5-11.5)^2 + (20-10)^2 + (20-10)^2) = sqrt(10^2 + 10^2 + 10^2) = sqrt(3 * 10^2) = 10 * sqrt(3)
        expected_cv3 = np.linalg.norm(com_ctd - com_bpc) # sqrt((21.5-31.5)^2 + (20-30)^2 + (20-30)^2) = sqrt((-10)^2 + (-10)^2 + (-10)^2) = sqrt(3 * 10^2) = 10 * sqrt(3)

        np.testing.assert_allclose(cvs, np.array([expected_cv1, expected_cv2, expected_cv3]), atol=1e-6)

        # Test with empty selections for domains
        empty_pdb_path = self.temp_path / "empty_protein.pdb"
        with open(empty_pdb_path, 'w') as f:
            f.write("ATOM    1  O   HOH W   1       1.000   1.000   1.000  1.00 0.00           O  \n")
            f.write("END\n")
        empty_universe = mda.Universe(str(empty_pdb_path))
        with self.assertRaises(ValueError):
            calculate_cvs(empty_universe) # Should raise error for empty CRBN-NTD

    def test_generate_fe_map_placeholder(self):
        grid = generate_fe_map_placeholder((10, 20, 30))
        self.assertIsInstance(grid, np.ndarray)
        self.assertEqual(grid.shape, (10, 20, 30))
        self.assertTrue(np.all(grid == 0))

        # Test invalid input
        with self.assertRaises(ValueError):
            generate_fe_map_placeholder((10, 20))
        with self.assertRaises(ValueError):
            generate_fe_map_placeholder((10, -1, 30))
        with self.assertRaises(ValueError):
            generate_fe_map_placeholder(("a", 10, 20))

    def test_snap_cv_to_fe_map_grid(self):
        # Example 1: Basic snapping
        cv_values = np.array([5.5, 12.0, 15.0])
        grid_dims = (10, 20, 30)
        grid_bounds = [(0.0, 10.0), (0.0, 20.0), (10.0, 30.0)]
        # Expected indices:
        # CV1: (5.5 - 0) / (10 - 0) = 0.55 -> floor(0.55 * 10) = 5
        # CV2: (12.0 - 0) / (20 - 0) = 0.60 -> floor(0.60 * 20) = 12
        # CV3: (15.0 - 10) / (30 - 10) = 5/20 = 0.25 -> floor(0.25 * 30) = 7
        expected_indices = (5, 12, 7)
        self.assertEqual(snap_cv_to_fe_map_grid(cv_values, grid_dims, grid_bounds), expected_indices)

        # Example 2: Values at boundaries
        cv_values_min = np.array([0.0, 0.0, 10.0])
        cv_values_max = np.array([10.0, 20.0, 30.0])
        self.assertEqual(snap_cv_to_fe_map_grid(cv_values_min, grid_dims, grid_bounds), (0, 0, 0))
        self.assertEqual(snap_cv_to_fe_map_grid(cv_values_max, grid_dims, grid_bounds), (9, 19, 29)) # Should snap to max index - 1

        # Example 3: Values outside boundaries (clamping)
        cv_values_below = np.array([-1.0, -5.0, 5.0])
        cv_values_above = np.array([11.0, 25.0, 35.0])
        self.assertEqual(snap_cv_to_fe_map_grid(cv_values_below, grid_dims, grid_bounds), (0, 0, 0))
        self.assertEqual(snap_cv_to_fe_map_grid(cv_values_above, grid_dims, grid_bounds), (9, 19, 29))

        # Example 4: Edge case where a value maps exactly to an integer, ensure it's not off by 1
        cv_values_exact = np.array([5.0, 10.0, 20.0]) # (5/10=0.5, 10/20=0.5, (20-10)/(30-10)=0.5)
        # Expected: floor(0.5*10)=5, floor(0.5*20)=10, floor(0.5*30)=15
        self.assertEqual(snap_cv_to_fe_map_grid(cv_values_exact, grid_dims, grid_bounds), (5, 10, 15))


        # Test invalid inputs
        with self.assertRaises(ValueError): # Wrong cv_values shape
            snap_cv_to_fe_map_grid(np.array([1.0, 2.0]), grid_dims, grid_bounds)
        with self.assertRaises(ValueError): # Wrong grid_dims length
            snap_cv_to_fe_map_grid(cv_values, (10, 20), grid_bounds)
        with self.assertRaises(ValueError): # Negative grid_dims
            snap_cv_to_fe_map_grid(cv_values, (10, -1, 30), grid_bounds)
        with self.assertRaises(ValueError): # Wrong grid_bounds length
            snap_cv_to_fe_map_grid(cv_values, grid_dims, [(0, 10), (0, 20)])
        with self.assertRaises(ValueError): # Invalid bound (min >= max)
            snap_cv_to_fe_map_grid(cv_values, grid_dims, [(0, 10), (0, 20), (10, 5)])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
