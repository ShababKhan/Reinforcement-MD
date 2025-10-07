import unittest
import numpy as np
import MDAnalysis as mda
import MDAnalysis.coordinates.PDB
from MDAnalysis.tests.datafiles import PDB_HOLE, DCD  # Example files if available
import os
import tempfile
from pathlib import Path

# Assume src is in the parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import (
    load_trajectory,
    extract_crbn_heavy_atom_coordinates,
    flatten_coordinates,
    superpose_trajectory_frames
)

class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up dummy PDB and DCD for testing."""
        cls.test_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.test_dir.name)

        # Create a simple PDB file for a 3-residue protein
        # PDB format requires specific spacing, so using f-strings carefully
        cls.dummy_pdb_path = cls.temp_path / "dummy_protein.pdb"
        with open(cls.dummy_pdb_path, 'w') as f:
            f.write("ATOM    1  N   ALA A   1      10.000  10.000  10.000  1.00 0.00           N  \n")
            f.write("ATOM    2  CA  ALA A   1      11.000  10.000  10.000  1.00 0.00           C  \n")
            f.write("ATOM    3  C   ALA A   1      12.000  10.000  10.000  1.00 0.00           C  \n")
            f.write("ATOM    4  O   ALA A   1      13.000  10.000  10.000  1.00 0.00           O  \n")
            f.write("ATOM    5  CB  ALA A   1      11.000  11.000  10.000  1.00 0.00           C  \n")

            f.write("ATOM    6  N   ALA A   2      20.000  20.000  20.000  1.00 0.00           N  \n")
            f.write("ATOM    7  CA  ALA A   2      21.000  20.000  20.000  1.00 0.00           C  \n")
            f.write("ATOM    8  C   ALA A   2      22.000  20.000  20.000  1.00 0.00           C  \n")
            f.write("ATOM    9  O   ALA A   2      23.000  20.000  20.000  1.00 0.00           O  \n")
            f.write("ATOM   10  CB  ALA A   2      21.000  21.000  20.000  1.00 0.00           C  \n")

            f.write("ATOM   11  N   GLY A   3      30.000  30.000  30.000  1.00 0.00           N  \n")
            f.write("ATOM   12  CA  GLY A   3      31.000  30.000  30.000  1.00 0.00           C  \n")
            f.write("ATOM   13  C   GLY A   3      32.000  30.000  30.000  1.00 0.00           C  \n")
            f.write("ATOM   14  O   GLY A   3      33.000  30.000  30.000  1.00 0.00           O  \n")
            f.write("TER\n") # Terminator for chain A
            f.write("END\n")

        # Create a simple DCD file with 2 frames
        cls.dummy_dcd_path = cls.temp_path / "dummy_trajectory.dcd"
        ref_universe = mda.Universe(str(cls.dummy_pdb_path))
        with mda.Writer(str(cls.dummy_dcd_path), n_atoms=ref_universe.n_atoms) as W:
            # Frame 1: Initial positions (from PDB)
            W.write(ref_universe.trajectory.next())
            # Frame 2: Shifted positions
            ref_universe.atoms.positions += 1.0 # Shift all atoms by 1A
            W.write(ref_universe.trajectory.next())

        # For CRBN length verification, we'll use a standard count for our dummy
        # In this dummy, we have 14 atoms. All are heavy protein atoms.
        # So, expected heavy atoms = 14, flattened length = 14 * 3 = 42.

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        cls.test_dir.cleanup()

    def test_load_trajectory(self):
        universe = load_trajectory(str(self.dummy_pdb_path), str(self.dummy_dcd_path))
        self.assertIsInstance(universe, mda.Universe)
        self.assertEqual(universe.n_atoms, 14)
        self.assertEqual(len(universe.trajectory), 2)

        # Test with non-existent files
        with self.assertRaises(IOError):
            load_trajectory("non_existent.pdb", "non_existent.dcd")

    def test_extract_crbn_heavy_atom_coordinates(self):
        universe = load_trajectory(str(self.dummy_pdb_path), str(self.dummy_dcd_path))
        universe.trajectory[0] # Ensure first frame is loaded

        coords = extract_crbn_heavy_atom_coordinates(universe)
        self.assertIsInstance(coords, np.ndarray)
        self.assertEqual(coords.shape, (14, 3)) # All 14 atoms are heavy protein atoms
        # Check a specific coordinate
        self.assertAlmostEqual(coords[0, 0], 10.000)

        # Test with no protein/heavy atoms (e.g. empty universe or selection)
        # Create a dummy universe with only water
        water_pdb_path = self.temp_path / "water.pdb"
        with open(water_pdb_path, 'w') as f:
            f.write("ATOM    1  O   HOH W   1       1.000   1.000   1.000  1.00 0.00           O  \n")
            f.write("ATOM    2  H1  HOH W   1       1.000   1.100   1.000  1.00 0.00           H  \n")
        water_universe = mda.Universe(str(water_pdb_path))
        coords_water = extract_crbn_heavy_atom_coordinates(water_universe)
        self.assertEqual(coords_water.shape, (0, 3)) # No heavy protein atoms
        self.assertTrue(coords_water.size == 0)

    def test_flatten_coordinates(self):
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        flattened = flatten_coordinates(coords)
        self.assertIsInstance(flattened, np.ndarray)
        self.assertEqual(flattened.shape, (9,))
        np.testing.assert_array_equal(flattened, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

        # Test empty input
        empty_coords = np.array([]).reshape(0, 3)
        flattened_empty = flatten_coordinates(empty_coords)
        self.assertEqual(flattened_empty.shape, (0,))
        np.testing.assert_array_equal(flattened_empty, np.array([]))

        # Test invalid shape
        with self.assertRaises(ValueError):
            flatten_coordinates(np.array([1, 2, 3])) # 1D
        with self.assertRaises(ValueError):
            flatten_coordinates(np.array([[1, 2], [3, 4]])) # N, 2

    def test_superpose_trajectory_frames(self):
        universe = load_trajectory(str(self.dummy_pdb_path), str(self.dummy_dcd_path))
        initial_frame_1_coords = universe.select_atoms('all').positions.copy()
        universe.trajectory[1] # Move to frame 2
        initial_frame_2_coords = universe.select_atoms('all').positions.copy()
        self.assertTrue(np.any(initial_frame_1_coords != initial_frame_2_coords)) # Verify frames are different

        # Superpose and get flattened coords
        superposed_universe, superposed_crbn_coords = superpose_trajectory_frames(universe)

        self.assertIs(superposed_universe, universe) # Check if universe is modified in place
        self.assertEqual(len(superposed_crbn_coords), 2)
        self.assertEqual(superposed_crbn_coords[0].shape, (14 * 3,)) # 14 heavy atoms
        self.assertEqual(superposed_crbn_coords[1].shape, (14 * 3,))

        # Verify that frame 0 is unchanged (it's the reference)
        universe.trajectory[0]
        np.testing.assert_allclose(universe.select_atoms('all').positions, initial_frame_1_coords)

        # Verify that frame 1 (originally shifted) is now superposed to frame 0
        universe.trajectory[1]
        # Calculate RMSD between superposed frame 1 and original frame 0 positions
        # The MDAnalysis superpose function shifts mobile_atoms positions
        # after alignment to the reference positions. So, the aligned frame should
        # be very close to the original reference frame coordinates.
        rmsd = mda.analysis.rms.rmsd(initial_frame_1_coords, universe.select_atoms('all').positions)
        self.assertLess(rmsd, 1e-6) # Should be very close after superposition

        # Check flattened coords
        flattened_frame0 = flatten_coordinates(initial_frame_1_coords)
        np.testing.assert_allclose(superposed_crbn_coords[0], flattened_frame0)
        np.testing.assert_allclose(superposed_crbn_coords[1], flattened_frame0, atol=1e-6) # Should be close to frame 0 after align

        # Test empty trajectory
        empty_pdb_path = self.temp_path / "empty.pdb"
        with open(empty_pdb_path, 'w') as f:
            f.write("END\n")
        empty_universe = mda.Universe(str(empty_pdb_path))
        empty_universe, empty_coords = superpose_trajectory_frames(empty_universe)
        self.assertEqual(len(empty_coords), 0)

        # Test with selection that yields no atoms
        universe_no_protein = mda.Universe(str(self.dummy_pdb_path))
        with self.assertRaises(ValueError):
            superpose_trajectory_frames(universe_no_protein, selection_string="resname LIG")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
