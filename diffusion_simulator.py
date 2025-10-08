import numpy as np

class DiffusionCoefficientCalculator:
    """
    Calculates the Diffusion Coefficient (D) based on Fick's First Law 
    for steady-state 1D diffusion.
    
    Fick's First Law: J = -D * (dC/dx)
    
    Attributes:
        tolerance (float): Tolerance for floating-point comparisons in tests.
    """
    
    def __init__(self, tolerance=1e-9):
        self.tolerance = tolerance

    def calculate_D(self, molar_flux_J: float, concentration_gradient_dCdx: float) -> float:
        """
        Calculates the Diffusion Coefficient (D) from the provided molar flux (J) 
        and concentration gradient (dC/dx).

        Args:
            molar_flux_J: The measured molar flux (in mol/(m^2*s)).
            concentration_gradient_dCdx: The concentration gradient (in mol/m^3 / m).

        Returns:
            The calculated Diffusion Coefficient D (in m^2/s).
            
        Raises:
            ValueError: If the gradient is zero and flux is non-zero, or vice-versa (unless handled).
        """
        if np.isclose(concentration_gradient_dCdx, 0.0, atol=self.tolerance):
            if not np.isclose(molar_flux_J, 0.0, atol=self.tolerance):
                # If gradient is zero, flux must be zero unless D is infinite, which is non-physical.
                raise ValueError("Flux (J) must be zero when concentration gradient (dC/dx) is zero.")
            return 0.0 # If both are zero, D can be considered zero or indeterminate, we return 0 for stability.

        # D = -J / (dC/dx)
        D = -molar_flux_J / concentration_gradient_dCdx
        return D

if __name__ == '__main__':
    # Example usage demonstration (not part of the final implemented module logic)
    calc = DiffusionCoefficientCalculator()
    
    # Test case: J = -10e-5 mol/(m^2*s), dC/dx = 0.01 mol/(m^3*m)
    # Expected D = -(-10e-5) / 0.01 = 10e-3 m^2/s = 0.001 m^2/s
    try:
        J = -10e-5
        dCdx = 0.01
        D_calc = calc.calculate_D(J, dCdx)
        print(f"Test Case: J={J}, dC/dx={dCdx}")
        print(f"Calculated D: {D_calc:.5f} m^2/s (Expected: 0.00100 m^2/s)")
    except ValueError as e:
        print(f"Error during example run: {e}")