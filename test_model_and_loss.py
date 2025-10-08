import unittest
import torch
import torch.nn as nn
from rmd_model import rMD_Network, INPUT_DIM, CV_DIM
from loss_functions import DualLoss, DEFAULT_W1, DEFAULT_W2

class TestModelAndLoss(unittest.TestCase):

    def setUp(self):
        """Set up dummy data and model instances for testing."""
        self.batch_size = 8
        self.model = rMD_Network()
        self.criterion = DualLoss(w1=2.0, w2=0.5) # Using custom weights for testing 
        
        # Dummy Input Data
        self.dummy_coords = torch.randn(self.batch_size, INPUT_DIM)
        # Dummy Target Data
        self.dummy_target_coords = torch.randn(self.batch_size, INPUT_DIM)
        self.dummy_target_cvs = torch.randn(self.batch_size, CV_DIM)

    # --- T2.1, T2.2, T2.3: Model Architecture Tests ---

    def test_model_forward_pass_dimensions(self):
        """
        T2.3 Acceptance Criteria: Verify that the rMD_Network returns the correct 
        output dimensions for LS and Pred.
        """
        ls_output, pred_output = self.model(self.dummy_coords)
        
        # LS output should be (Batch Size, CV_DIM) -> (8, 3)
        self.assertEqual(ls_output.shape, (self.batch_size, CV_DIM), "Latent Space output has incorrect dimensions.")
        
        # Prediction output should be (Batch Size, INPUT_DIM) -> (8, 9696)
        self.assertEqual(pred_output.shape, (self.batch_size, INPUT_DIM), "Prediction output has incorrect dimensions.")

    def test_model_layers_exist(self):
        """
        T2.1/T2.2 Verification: Check if the Encoder and Decoder networks 
        contain the expected number of layers (5 Linear layers each).
        """
        # Encoder (5 Linear layers + 4 Swish activations)
        encoder_linear_layers = [module for module in self.model.encoder.modules() if isinstance(module, nn.Linear)]
        # Decoder (5 Linear layers + 4 Swish activations)
        decoder_linear_layers = [module for module in self.model.decoder.modules() if isinstance(module, nn.Linear)]
        
        self.assertEqual(len(encoder_linear_layers), 5, "Encoder must have 5 Linear layers.")
        self.assertEqual(len(decoder_linear_layers), 5, "Decoder must have 5 Linear layers.")

    # --- T3.1, T3.2, T3.3: Dual-Loss Function Tests ---

    def test_dual_loss_output_types_and_values(self):
        """
        T3.3 Verification: Check if DualLoss produces a scalar loss and if all 
        individual losses are positive scalars.
        """
        ls_output, pred_output = self.model(self.dummy_coords)
        
        total_loss, loss1, loss2 = self.criterion(
            ls_output, pred_output, self.dummy_target_cvs, self.dummy_target_coords
        )
        
        # Check if Total Loss is a single scalar tensor
        self.assertTrue(total_loss.dim() == 0, "Total loss must be a scalar tensor.")
        
        # Check if individual losses are positive (as is typical for loss functions)
        self.assertTrue(loss1.item() >= 0, "Loss1 (Latent) must be non-negative.")
        self.assertTrue(loss2.item() >= 0, "Loss2 (Reconstruction) must be non-negative.")

    def test_weighted_loss_calculation(self):
        """
        T3.3 Verification: Manually calculate a simple loss case to ensure 
        the weighted sum formula is correct.
        """
        # Create trivial inputs where Loss1 = 1.0 and Loss2 = 2.0
        # This bypasses the model complexities; we only test the loss function arithmetic.
        
        # Case 1: Loss1 (LS vs CVs) = 1.0 (difference 1.0)
        test_ls = torch.ones(self.batch_size, CV_DIM) * 0.0
        test_cvs = torch.ones(self.batch_size, CV_DIM) * 1.0
        
        test_loss1_val = nn.MSELoss(reduction='mean')(test_ls, test_cvs).item() 
        # MSE for difference of 1.0 is 1.0

        # Case 2: Loss2 (Pred vs Target) = 2.0 (mean absolute difference 2.0)
        test_pred = torch.ones(self.batch_size, INPUT_DIM) * 0.0
        test_target = torch.ones(self.batch_size, INPUT_DIM) * 2.0
        
        # L1 Loss (MAE) is the mean absolute difference, which is 2.0
        test_loss2_val = nn.L1Loss(reduction='mean')(test_pred, test_target).item() 
        # MAE for difference of 2.0 is 2.0
        
        # Test the DualLoss function with these known, simplified values:
        # Weights used in setup: W1=2.0, W2=0.5
        # Expected Total Loss = (2.0 * 1.0) + (0.5 * 2.0) = 2.0 + 1.0 = 3.0
        
        total_loss, _, _ = self.criterion(
            ls_coords=test_ls, 
            pred_coords=test_pred, 
            target_cvs=test_cvs, 
            target_coords=test_target
        )
        
        # Check the final calculated weighted loss
        self.assertAlmostEqual(total_loss.item(), 3.0, places=5, msg="Weighted total loss calculation incorrect.")
        
    def test_rmsd_calculation(self):
        """
        Verification of the static RMSD utility function for Loss2 metric reporting.
        """
        # Test Case: Two points differ by exactly 1 unit in each direction.
        # N_coords = 6. (2 atoms, 3 dimensions, reduced to 6 for easy calculation)

        # Flat: [0, 0, 0, 0, 0, 0]
        pred = torch.zeros(1, 6) 
        # Flat: [1, 1, 1, 1, 1, 1]
        target = torch.ones(1, 6) 
        
        # Total distance squared = 6 * (1-0)^2 = 6
        # RMSD = sqrt( Total Distance Squared / N_coords )
        # RMSD = sqrt( 6 / 6 ) = sqrt(1) = 1.0
        
        # Temporarily adapt the input dimensions to match the simplified test case
        # (The function assumes the last dimension is the coordinate count)
        original_input_dim = INPUT_DIM
        rmsd = DualLoss.calculate_rmsd(pred, target)
        
        self.assertAlmostEqual(rmsd.item(), 1.0, places=5, msg="RMSD calculation failed for simple case.")

if __name__ == '__main__':
    unittest.main()