import unittest
import numpy as np
from diffusion_estimator import calculate_diffusion_coefficient

class TestDiffusionEstimator(unittest.TestCase):
    
    def test_basic_linear_profile(self):
        # Test case: Simple linear increase in D(C)
        # C0 = 10. L is irrelevant for the integral check unless boundary conditions change.
        # Data: (C, D): [(1, 10), (2, 20), (3, 30)]. Linear D(C) = 10*C
        # Integral of 10*C dC from 1 to 3 is: [5*C^2] from 1 to 3
        # = 5*(9 - 1) = 40.
        C0 = 10.0
        gradient_data = [
            (1.0, 10.0),  # C=1, D=10
            (2.0, 20.0),  # C=2, D=20
            (3.0, 30.0)   # C=3, D=30
        ]
        # Expected D_eff = (1 / (2 * C0)) * Integral = (1 / 20) * 40 = 2.0
        expected_D_eff = 2.0
        
        result_D_eff = calculate_diffusion_coefficient(1.0, C0, gradient_data)
        self.assertAlmostEqual(result_D_eff, expected_D_eff, places=5)

    def test_zero_c0_raises_error(self):
        # Edge Case 1: C0 is zero
        gradient_data = [(1.0, 10.0), (2.0, 20.0)]
        with self.assertRaisesRegex(ValueError, "C0 cannot be zero"):
            calculate_diffusion_coefficient(1.0, 0.0, gradient_data)

    def test_insufficient_data_raises_error(self):
        # Edge Case 2: Less than two data points
        with self.assertRaisesRegex(ValueError, "fewer than two points"):
            calculate_diffusion_coefficient(1.0, 10.0, [(1.0, 10.0)])

    def test_non_uniform_spacing(self):
        # Test case with non-uniform spacing (Trapezoidal rule necessity)
        # Data points: (C, D)
        # 1. C=1 -> D=5. Segment width 1. Integral approx: 0.5 * (5 + 15) * 1 = 10
        # 2. C=1.5 -> D=15. Segment width 0.5. Integral approx: 0.5 * (15 + 20) * 0.5 = 8.75
        # 3. C=2.5 -> D=20. Total Integral approx: 10 + 8.75 = 18.75
        C0 = 5.0
        gradient_data = [
            (1.0, 5.0),
            (1.5, 15.0),
            (3.0, 20.0)
        ]
        # Expected D_eff = (1 / (2 * 5.0)) * 18.75 = 1.875
        expected_D_eff = 1.875
        
        result_D_eff = calculate_diffusion_coefficient(1.0, C0, gradient_data)
        self.assertAlmostEqual(result_D_eff, expected_D_eff, places=5)

if __name__ == '__main__':
    unittest.main()