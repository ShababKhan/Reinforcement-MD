import unittest
import numpy as np
from rmd_data_preprocessing import flatten_coordinates, superpose_structures


class TestRMDDataPreprocessing(unittest.TestCase):

    def test_flatten_coordinates_valid_input(self): 
        """Test flatten_coordinates with a typical 2D array."""
        coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_flat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = flatten_coordinates(coords)
        self.assertTrue(np.array_equal(result, expected_flat))
        self.assertEqual(result.shape, (9,))

    def test_flatten_coordinates_single_atom(self):
        """Test flatten_coordinates with a single atom."""
        coords = np.array([[10, 20, 30]])
        expected_flat = np.array([10, 20, 30])
        result = flatten_coordinates(coords)
        self.assertTrue(np.array_equal(result, expected_flat))
        self.assertEqual(result.shape, (3,))

    def test_flatten_coordinates_empty_input(self):
        """Test flatten_coordinates with an empty array."""
        coords = np.empty((0, 3))
        expected_flat = np.empty((0,))
        result = flatten_coordinates(coords)
        self.assertTrue(np.array_equal(result, expected_flat))
        self.assertEqual(result.shape, (0,))

    def test_flatten_coordinates_invalid_ndim(self):
        """Test flatten_coordinates with incorrect number of dimensions."""
        coords_3d = np.array([[[1, 2, 3], [4, 5, 6]]])
        coords_1d = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            flatten_coordinates(coords_3d)
        with self.assertRaises(ValueError):
            flatten_coordinates(coords_1d)

    def test_flatten_coordinates_invalid_last_dim(self):
        """Test flatten_coordinates with incorrect last dimension size."""
        coords = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            flatten_coordinates(coords)

    def test_flatten_coordinates_non_numpy_input(self):
        """Test flatten_coordinates with non-NumPy input."""
        with self.assertRaises(TypeError):
            flatten_coordinates([[1, 2, 3]])

    def test_superpose_structures_identical(self):
        """Test superposition of an identical structure."""
        ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        structures = np.array([ref])
        superposed = superpose_structures(structures, ref)
        self.assertTrue(np.allclose(superposed[0], ref))

    def test_superpose_structures_translated(self):
        """Test superposition of a translated structure."""
        ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        translated = ref + np.array([10, 20, 30])
        structures = np.array([translated])
        superposed = superpose_structures(structures, ref)
        self.assertTrue(np.allclose(superposed[0], ref))

    def test_superpose_structures_rotated(self):
        """Test superposition of a rotated structure."""
        ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        # Rotate by 90 degrees around Z-axis
        rotated_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotated = ref @ rotated_matrix
        structures = np.array([rotated])
        superposed = superpose_structures(structures, ref)
        self.assertTrue(np.allclose(superposed[0], ref, atol=1e-7))

    def test_superpose_structures_rotated_translated(self):
        """Test superposition of a rotated and translated structure."""
        ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        # Rotate by 90 degrees around Z-axis
        rotated_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotated_translated = (ref @ rotated_matrix) + np.array([10, 20, 30])
        structures = np.array([rotated_translated])
        superposed = superpose_structures(structures, ref)
        self.assertTrue(np.allclose(superposed[0], ref, atol=1e-7))

    def test_superpose_structures_multiple_frames(self):
        """Test superposition with multiple frames."""
        ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        translated = ref + np.array([1, 1, 1])
        rotated_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotated = ref @ rotated_matrix
        mixed_structures = np.array([ref, translated, rotated])
        expected_superposed = np.array([ref, ref, ref]) # All should align to ref

        superposed = superpose_structures(mixed_structures, ref)
        self.assertTrue(np.allclose(superposed, expected_superposed, atol=1e-7))

    def test_superpose_structures_mismatched_atom_count(self):
        """Test superposition with mismatched atom counts."""
        ref = np.array([[0, 0, 0], [1, 0, 0]])
        structures = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]) # 3 atoms in structure, 2 in ref
        with self.assertRaises(ValueError):
            superpose_structures(structures, ref)

    def test_superpose_structures_invalid_ref_shape(self):
        """Test superposition with invalid reference structure shape."""
        ref_1d = np.array([0, 0, 0])
        ref_3d = np.array([[[0, 0, 0]]])
        structures = np.array([[[0, 0, 0]]])
        with self.assertRaises(ValueError):
            superpose_structures(structures, ref_1d)
        with self.assertRaises(ValueError):
            superpose_structures(structures, ref_3d)

    def test_superpose_structures_invalid_structures_shape(self):
        """Test superposition with invalid input structures shape."""
        ref = np.array([[0, 0, 0]])
        structures_2d = np.array([[0, 0, 0]])
        structures_4d = np.array([[[[0, 0, 0]]]])
        with self.assertRaises(ValueError):
            superpose_structures(structures_2d, ref)
        with self.assertRaises(ValueError):
            superpose_structures(structures_4d, ref)

    def test_superpose_structures_non_numpy_input(self):
        """Test superposition with non-NumPy input."""
        ref = np.array([[0, 0, 0]])
        structures = np.array([[[0, 0, 0]]])
        with self.assertRaises(TypeError):
            superpose_structures([[0, 0, 0]], ref)
        with self.assertRaises(TypeError):
            superpose_structures(structures, [[0, 0, 0]])

    def test_superpose_structures_single_atom_aligned(self):
        """Test superposition with single-atom structures."""
        ref = np.array([[1.0, 2.0, 3.0]])
        structure = np.array([[10.0, 20.0, 30.0]])
        structures_input = np.array([structure])
        superposed = superpose_structures(structures_input, ref)
        self.assertTrue(np.allclose(superposed[0], ref, atol=1e-7))

    def test_superpose_structures_linear_data(self):
        """Test superposition with degenerate (linear) point sets."""
        ref = np.array([[0,0,0], [1,0,0], [2,0,0]])
        struct = np.array([[1,1,1], [2,1,1], [3,1,1]])
        structures_input = np.array([struct])
        superposed = superpose_structures(structures_input, ref)
        self.assertTrue(np.allclose(superposed[0], ref, atol=1e-7))
