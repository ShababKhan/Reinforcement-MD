import unittest
import numpy as np
import os

# Assuming data_utils.py is accessible in the environment. 
# Import the actual functions and constants needed for testing.
from data_utils import generate_synthetic_data, superpose_and_flatten, INPUT_DIM, CV_DIM, N_FRAMES

class TestDataUtils(unittest.TestCase):

    def test_synthetic_data_generation_dimensions(self):
        """
        Verify that generate_synthetic_data produces arrays with the correct dimensions.
        (Matches T1.2 Acceptance Criteria: (N_FRAMES, INPUT_DIM) and (N_FRAMES, CV_DIM))
        """
        coords, cvs = generate_synthetic_data()

        # Check Frames
        self.assertEqual(coords.shape[0], N_FRAMES)
        self.assertEqual(cvs.shape[0], N_FRAMES)

        # Check Dimensions
        self.assertEqual(coords.shape[1], INPUT_DIM)
        self.assertEqual(cvs.shape[1], CV_DIM)

    def test_synthetic_data_is_numpy_array(self):
        """
        Verify that the output data types are NumPy arrays.
        """
        # Ensure we are not testing against default values 
        coords, cvs = generate_synthetic_data(n_frames=10, n_atoms=10) 
        self.assertIsInstance(coords, np.ndarray)
        self.assertIsInstance(cvs, np.ndarray)

    def test_superposition_removes_global_translation(self):
        """
        Verify that the superposition function centers the coordinates 
        (Center of Geometry at origin). This verifies correct T1.3 preprocessing logic.
        """
        # Create dummy 3D coordinate data with clear translational offsets
        test_coords_3d = np.array([
            # Frame 0: Both atoms shifted by [10, 5, 0]. COG is [10, 5, 0].
            [[10.0-0.5, 5.0-0.5, 0.0-0.5], [10.0+0.5, 5.0+0.5, 0.0+0.5]],
            # Frame 1: Both atoms shifted by [1, 1, 1]. COG is [1, 1, 1].
            [[1.0-0.5, 1.0-0.5, 1.0-0.5], [1.0+0.5, 1.0+0.5, 1.0+0.5]]
        ]) 
        
        coords_flat = superpose_and_flatten(test_coords_3d)
        
        # 1. Un-flatten into (N_frames, N_atoms, 3)
        n_test_frames, n_atoms_flat = coords_flat.shape
        n_atoms = int(n_atoms_flat / 3)
        superposed_coords_3d = coords_flat.reshape(coords_flat.shape[0], n_atoms, 3)

        # 2. Calculate the COG for the resulting superposed frames
        cog_frame_0 = np.mean(superposed_coords_3d[0], axis=0)
        cog_frame_1 = np.mean(superposed_coords_3d[1], axis=0)

        # The COG should be approximately [0, 0, 0] after centering (superposition)
        # Using a small tolerance (atol=1e-8) for floating point checks
        np.testing.assert_allclose(cog_frame_0, [0, 0, 0], atol=1e-8, err_msg="Frame 0 COG was not centered.")
        np.testing.assert_allclose(cog_frame_1, [0, 0, 0], atol=1e-8, err_msg="Frame 1 COG was not centered.")

    def test_flattening_correctness(self):
        """
        Verify that the final output from superposition is correctly flattened (T1.3).
        """
        # Generate simple data for a quick integrity check (2 frames, 2 atoms)
        test_coords_3d = np.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
        ])
        
        coords_flat = superpose_and_flatten(test_coords_3d)
        
        # Expected shape is (2 frames, 2 atoms * 3 coords = 6)
        expected_flat_shape = (2, 6)
        self.assertEqual(coords_flat.shape, expected_flat_shape)
        
        # Check that the number of columns in the 2D array is N_atoms * 3
        self.assertEqual(coords_flat.shape[1], test_coords_3d.shape[1] * 3)

if __name__ == '__main__':
    # Due to the constraints of the environment, saving the test file outside the 
    # intended 'tests/' directory for a single call.
    unittest.main()
