"""Unit tests for the models module and force_provider module."""
import unittest
import torch
import numpy as np
from pathlib import Path
from abc import ABCMeta
import sys

# Attempt to configure imports for both running locally and via script
try:
    from src.models import PhysicsInformedAutoencoder
    from src.force_provider import AbstractForceProvider, MockForceProvider
    from src.md_engine import SystemState
except ImportError:
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir.parent / "src"))
    from models import PhysicsInformedAutoencoder
    from force_provider import AbstractForceProvider, MockForceProvider
    from md_engine import SystemState


# --- US 1.3 Tests (Models) ---
class TestPhysicsInformedAutoencoder(unittest.TestCase):
    """
    Tests the structural integrity and dimensionality of the PI-AE model.
    """
    def setUp(self):
        """Set up standard parameters for the PI-AE model."""
        self.input_dim = 6
        self.latent_dim = 2
        
        # Test with the default architecture: [32, 16]
        self.model = PhysicsInformedAutoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        )
        
    def test_model_initialization(self):
        """Verify that the model initializes without raising errors."""
        self.assertIsInstance(self.model, PhysicsInformedAutoencoder)
        
        # Verify the presence of encoder and decoder components
        self.assertTrue(hasattr(self.model, 'encoder'))
        self.assertTrue(hasattr(self.model, 'decoder'))

    def test_forward_pass_dims(self):
        """
        Verify that a forward pass yields output tensors with the correct dimensions.
        """
        batch_size = 10
        # Simulated input data (Batch x Input_Dim)
        x = torch.randn(batch_size, self.input_dim)
        
        reconstruction, latent = self.model.forward(x)
        
        # 1. Reconstruction output must match the input dimension
        self.assertEqual(reconstruction.shape, (batch_size, self.input_dim), 
                         "Reconstruction output shape is incorrect.")
        
        # 2. Latent output must match the defined latent dimension
        self.assertEqual(latent.shape, (batch_size, self.latent_dim),
                         "Latent space output shape is incorrect.")

    def test_latent_space_method_dims(self):
        """
        Verify that the get_latent_space convenience method returns the correct dimensions.
        """
        batch_size = 5
        x = torch.randn(batch_size, self.input_dim)
        
        latent = self.model.get_latent_space(x)

        # Latent output must match the defined latent dimension
        self.assertEqual(latent.shape, (batch_size, self.latent_dim),
                         "get_latent_space method output shape is incorrect.")


# --- US 2.3 Tests (Force Provider) ---
class TestForceProvider(unittest.TestCase):
    """
    Tests the structural integrity and functionality of the Force Provider interface.
    """
    
    def test_abstract_force_provider(self):
        """Verify that AbstractForceProvider cannot be instantiated."""
        # Using ABCMeta directly for a robust check
        self.assertIsInstance(AbstractForceProvider, ABCMeta)
        with self.assertRaises(TypeError):
            AbstractForceProvider()

    def test_mock_force_provider_output(self):
        """Verify the MockForceProvider returns the expected force shape and value."""
        num_atoms = 5
        mock_f_value = 1.0 
        provider = MockForceProvider(num_atoms, mock_f_value)
        state = SystemState(num_atoms) # Requires SystemState from md_engine

        force = provider.calculate_force(state)

        # Check shape (N, 3)
        self.assertEqual(force.shape, (num_atoms, 3))
        
        # Check that the specific mock force was applied
        expected_force = np.zeros((num_atoms, 3))
        expected_force[0, 0] = mock_f_value
        
        np.testing.assert_array_almost_equal(force, expected_force)

    def test_mock_potential_energy(self):
        """Verify the MockForceProvider returns the expected constant potential energy."""
        provider = MockForceProvider(1)
        state = SystemState(1)
        
        energy = provider.get_potential_energy(state)
        
        # Expected energy is 0.0 as defined in the mock class
        self.assertEqual(energy, 0.0)
