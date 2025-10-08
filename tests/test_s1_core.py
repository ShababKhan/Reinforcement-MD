import unittest
import numpy as np
import torch

# Import modules created in Sprint 1 tasks
from src.data_utils import load_r_batch_data, generate_mock_structure_data, superpose_structures, STRUCTURE_DIMENSION, CV_DIMENSION
from src.losses import calculate_loss2_rmsd, calculate_loss1_latent_cv_rmsd, DualLoss

class TestDataUtils(unittest.TestCase):
    """Unit tests for S1.T1: Data Ingestion & Preprocessing."""

    def test_mock_data_generation(self):
        """Verify mock data generation produces correct shapes."""
        X_raw = generate_mock_structure_data(n_samples=100)
        self.assertEqual(X_raw.shape, (100, STRUCTURE_DIMENSION))
        
        Y_cv = load_r_batch_data()[1] # Load CV data only
        self.assertEqual(Y_cv.shape[1], CV_DIMENSION)
        self.assertEqual(Y_cv.shape[0], 8000) # Check default size for training set scope

    def test_superposition_centering(self):
        """Verify the simulation of structure superposition (centering)."""
        # Create data where the mean is intentionally non-zero
        data = np.array([
            [10.0, 10.0, 10.0],
            [12.0, 12.0, 12.0],
            [ 8.0,  8.0,  8.0]
        ], dtype=np.float32)
        
        # Reference frame is index 0 -> mean is 10.0
        superposed = superpose_structures(data, reference_frame_index=0)
        
        # In our simulation, the entire batch is centered around the reference mean (10.0)
        # Expected result: [0.0, 2.0, -2.0] relative to the reference *mean*
        self.assertTrue(np.allclose(superposed.mean(axis=0), 0.0))
        
        # Individual centering check:
        # Original mean = 10.0
        # Element 1 (10.0) -> 10.0 - 10.0 = 0.0
        # Element 2 (12.0) -> 12.0 - 10.0 = 2.0
        # Element 3 (8.0) -> 8.0 - 10.0 = -2.0
        expected = np.array([0.0, 2.0, -2.0], dtype=np.float32)
        self.assertTrue(np.allclose(superposed[0], expected))
        
    def test_load_r_batch_data_shape(self):
        """Verify the main data loading function returns the correct shapes for the split."""
        X, Y_cv = load_r_batch_data()
        self.assertEqual(X.shape[0], 8000) # Training set size
        self.assertEqual(X.shape[1], STRUCTURE_DIMENSION)
        self.assertEqual(Y_cv.shape[0], 8000)
        self.assertEqual(Y_cv.shape[1], CV_DIMENSION)

class TestLossFunctions(unittest.TestCase):
    """Unit tests for S1.T4, S1.T5: Loss 1 and Loss 2 Calculations."""

    def setUp(self):
        """Setup common tensors for loss testing."""
        self.batch_size = 16
        self.feature_dim = 9696
        self.ls_dim = 3
        
        # Mock prediction/target pairs for Loss 2 (Reconstruction)
        self.target = torch.randn(self.batch_size, self.feature_dim)
        # Prediction is identical to target (Loss2 should be ~0)
        self.perfect_pred = self.target.clone() 
        # Prediction has a fixed error of 0.1 for all coordinates (Loss2 should be ~0.01)
        self.error_pred = self.target + 0.1
        
        # Mock LS/CV pairs for Loss 1 (Physics correlation)
        self.perfect_ls = torch.randn(self.batch_size, self.ls_dim)
        self.perfect_cv = self.perfect_ls.clone()
        self.error_ls = self.perfect_ls + 1.0
        
    def test_loss2_perfect(self):
        """Loss 2 should be zero if prediction matches target exactly."""
        loss = calculate_loss2_rmsd(self.perfect_pred, self.target)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_loss2_fixed_error(self):
        """Loss 2 (MSE) should equal the square of the fixed error (0.1^2 = 0.01)."""
        loss = calculate_loss2_rmsd(self.error_pred, self.target)
        self.assertAlmostEqual(loss.item(), 0.01, places=6)
        
    def test_loss1_perfect(self):
        """Loss 1 should be zero if LS coordinates match CV targets exactly."""
        loss = calculate_loss1_latent_cv_rmsd(self.perfect_ls, self.perfect_cv)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_loss1_fixed_error(self):
        """Loss 1 (MSE) should equal the square of the fixed error (1.0^2 = 1.0)."""
        loss = calculate_loss1_latent_cv_rmsd(self.error_ls, self.perfect_cv)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_dual_loss_combination(self):
        """Verify the weighted sum in the DualLoss class."""
        TOLERANCE = 1.0 # Allow large tolerance given mock data scaling
        
        # Setup perfect L2 and L1 (should result in L1 weight)
        # L2 = 0, L1 = 0 -> Total = 0
        total_loss, _, _ = DualLoss(alpha=0.5).forward(self.perfect_pred, self.target, self.perfect_ls, self.perfect_cv)
        self.assertAlmostEqual(total_loss.item(), 0.0, places=6)
        
        # Setup L2 = 0.01, L1 = 1.0, alpha = 0.5
        # Expected Total = 0.5 * 1.0 + 0.01 = 0.51
        expected_total = 0.51
        total_loss, _, _ = DualLoss(alpha=0.5).forward(self.error_pred, self.target, self.error_ls, self.perfect_cv)
        self.assertAlmostEqual(total_loss.item(), expected_total, places=4)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)