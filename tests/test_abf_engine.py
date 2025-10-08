"""Unit tests for the abf_engine module (ABFHistogram)."""
import unittest
import numpy as np
from pathlib import Path
import sys

# Import the class to be tested
try:
    from src.abf_engine import ABFHistogram
except ImportError:
    # Adjust path if running outside the correct env setup
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir.parent / "src"))
    from abf_engine import ABFHistogram


class TestABFHistogram(unittest.TestCase):
    """
    Tests the fundamental binning, accumulation, and mean force calculation 
    of the ABFHistogram class (US 3.2 Scientific Validation).
    """
    def setUp(self):
        """Standard setup for a 2D histogram."""
        self.bounds = [(0.0, 10.0), (-5.0, 5.0)] # xi1 from 0 to 10, xi2 from -5 to 5
        self.bins = [10, 5]                       # 10 bins for xi1, 5 bins for xi2
        self.dim = 2
        self.hist = ABFHistogram(self.bounds, self.bins)

    def test_initialization_and_dimensions(self):
        """Verify the internal arrays are correctly initialized."""
        self.assertEqual(self.hist.dim, 2)
        
        # Counts shape should be (10, 5)
        self.assertEqual(self.hist.counts.shape, tuple(self.bins))
        
        # Sum_Forces shape should be (10, 5, 2) - [SCIENTIFIC BENCHMARK: Dimensional Integrity]
        self.assertEqual(self.hist.sum_forces.shape, (10, 5, self.dim))
        
        # Check bin widths
        np.testing.assert_array_almost_equal(self.hist.bin_widths, [1.0, 2.0])

    def test_bin_indexing_in_bounds(self):
        """Test correct conversion of continuous xi to discrete bin indices."""
        # xi1 = 0.5 (Bin 0), xi2 = 0.0 (Bin 2, since [-5, -3) is 0, [-3, -1) is 1, [-1, 1) is 2)
        xi_in = np.array([0.5, 0.0])
        expected_indices = np.array([0, 2])
        
        indices = self.hist._get_bin_indices(xi_in)
        np.testing.assert_array_equal(indices, expected_indices)

        # Test index at upper boundary (should go into the last bin)
        xi_upper = np.array([9.9, 4.9]) # Close to max
        expected_indices_upper = np.array([9, 4])
        
        indices_upper = self.hist._get_bin_indices(xi_upper)
        np.testing.assert_array_equal(indices_upper, expected_indices_upper)

    def test_bin_indexing_out_of_bounds(self):
        """Test that points outside bounds return the -1 sentinel value."""
        xi_low = np.array([-0.1, 0.0])
        xi_high = np.array([10.1, 0.0])
        xi_mixed = np.array([5.0, 6.0])

        expected_sentinel = np.array([-1, -1])
        
        np.testing.assert_array_equal(self.hist._get_bin_indices(xi_low), expected_sentinel)
        np.testing.assert_array_equal(self.hist._get_bin_indices(xi_high), expected_sentinel)
        np.testing.assert_array_equal(self.hist._get_bin_indices(xi_mixed), expected_sentinel)

    def test_accumulation_logic(self):
        """Test that count and total force are accumulated correctly."""
        xi_sample = np.array([1.5, -2.0]) # Bin indices [1, 1]
        F_sample_1 = np.array([10.0, 5.0])
        F_sample_2 = np.array([20.0, -5.0])
        
        # Accumulate first sample
        self.hist.accumulate_sample(xi_sample, F_sample_1)
        
        self.assertEqual(self.hist.counts[1, 1], 1)
        
        # Accumulate second sample in the same bin
        self.hist.accumulate_sample(xi_sample, F_sample_2)
        
        # Check state after 2 samples
        self.assertEqual(self.hist.counts[1, 1], 2)
        np.testing.assert_array_equal(self.hist.sum_forces[1, 1], np.array([30.0, 0.0]))

    def test_bias_force_calculation(self):
        """
        [SCIENTIFIC BENCHMARK] Test that the instantaneous bias force is calculated as -MeanForce.
        """
        xi_bin = np.array([5.5, 3.0]) # Bin indices [5, 4]
        
        # Input: Total F_sum = [-2.0, 0.0], Count = 2
        self.hist.accumulate_sample(xi_bin, np.array([1.0, 1.0]))
        self.hist.accumulate_sample(xi_bin, np.array([-3.0, -1.0]))
        
        # Mean Force $\langle F \rangle = [-2.0/2, 0.0/2] = [-1.0, 0.0]$
        # Bias Force $F_{bias} = -\langle F \rangle = [1.0, 0.0]$
        
        F_bias = self.hist.get_instantaneous_bias(xi_bin)
        F_expected = np.array([1.0, 0.0])
        
        np.testing.assert_array_almost_equal(F_bias, F_expected)

    def test_bias_force_zero_count(self):
        """Test that zero force is returned if the bin is empty."""
        xi_empty = np.array([0.5, 0.0]) # Bin [0, 2] is empty
        F_bias = self.hist.get_instantaneous_bias(xi_empty)
        F_expected = np.array([0.0, 0.0])
        
        np.testing.assert_array_equal(F_bias, F_expected)
