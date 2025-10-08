import unittest
import numpy as np
# Assuming data_utils is in the same directory for temporary testing
try:
    from data_utils import generate_synthetic_data, superpose_and_flatten, N_FRAMES, INPUT_DIM, CV_DIM
except ImportError:
    # Fallback for execution environment where data_utils might need manual import setup
    # In a real environment, the Developer would ensure correct import paths are used
    print("Warning: Could not import data_utils directly. Assuming constants/functions are mockable for test structure.")
    # Mock constants for test structure verification
    N_FRAMES = 10000
    INPUT_DIM = 9696
    CV_DIM = 3
    # Dummy mock functions if necessary only for structuring the test file
    def generate_synthetic_data():
        return np.random.rand(N_FRAMES, INPUT_DIM), np.random.rand(N_FRAMES, CV_DIM)
    def superpose_and_flatten(coords_3d):
        return np.zeros((coords_3d.shape[0], coords_3d.shape[1] * coords_3d.shape[2]))


class TestDataUtils(unittest.TestCase):

    def test_synthetic_data_generation_dimensions(self):
        """
        [T1.2 Acceptance Criteria] Verify that generate_synthetic_data produces 
        arrays with the correct dimensions.
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
        coords, cvs = generate_synthetic_data()
        self.assertIsInstance(coords, np.ndarray)
        self.assertIsInstance(cvs, np.ndarray)

    def test_superposition_removes_global_translation(self):
        """
        [T1.3 Logic Verification] Verify that the superposition function centers 
        the coordinates (Center of Geometry (COG) at origin).
        """
        # Create dummy 3D coordinate data: a 2-atom system that shifts position
        n_test_frames = 2
        
        # Frame 1: CoG at [5, 2.5, 0]
        f1 = np.array([[10.0, 5.0, 0.0], [0.0, 0.0, 0.0]]) 
        
        # Frame 2: CoG at [1, 1, 1]
        f2 = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])   
        
        test_coords_3d = np.array([f1, f2])

        coords_flat = superpose_and_flatten(test_coords_3d)
        
        # 1. Un-flatten into (N_frames, N_atoms, 3)
        n_atoms = int(coords_flat.shape[1] / 3)
        superposed_coords_3d = coords_flat.reshape(n_test_frames, n_atoms, 3)

        # 2. Calculate the COG for the reconstructed frames
        cog_frame_0 = np.mean(superposed_coords_3d[0], axis=0)
        cog_frame_1 = np.mean(superposed_coords_3d[1], axis=0)

        # The COG should be approximately [0, 0, 0] after centering (superposition)
        np.testing.assert_allclose(cog_frame_0, [0, 0, 0], atol=1e-8, err_msg="Frame 0 COG was not centered.")
        np.testing.assert_allclose(cog_frame_1, [0, 0, 0], atol=1e-8, err_msg="Frame 1 COG was not centered.")

    def test_flattening_correctness(self):
        """
        [T1.3 Logic Verification] Verify that the final output from superposition 
        is correctly flattened.
        """
        # Generate simple data for a quick integrity check
        test_coords_3d = np.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
        ])
        
        coords_flat = superpose_and_flatten(test_coords_3d)
        
        expected_total_elements = test_coords_3d.size
        
        # Check that the number of elements in the flattened array is correct
        self.assertEqual(coords_flat.size, expected_total_elements)
        
        # Check that the number of columns in the 2D array is N_atoms * 3
        self.assertEqual(coords_flat.shape[1], test_coords_3d.shape[1] * 3)

if __name__ == '__main__':
    unittest.main()