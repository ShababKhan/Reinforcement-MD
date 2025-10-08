import numpy as np
from typing import List, Tuple

def calculate_diffusion_coefficient(
    L: float, 
    C0: float, 
    gradient_data: List[Tuple[float, float]]
) -> float:
    """
    Calculates the effective diffusion coefficient (D_eff) by integrating
    the concentration profile according to the Fickian diffusion model
    for a discrete measurement set.

    The calculation implements the integral form: D_eff = (1 / (2 * C0)) * integral(D(C)),
    where the integral is approximated using the Trapezoidal Rule on the
    provided (C, D) pairs. This function assumes the input gradient_data 
    provides pairs of (Concentration C, Local Diffusion D(C)).

    @param L: The macroscopic length scale over which diffusion occurs (m).
    @param C0: The initial or reference concentration (mol/m^3).
    @param gradient_data: A list of tuples, where each tuple is (C, D(C)).
                          C is the concentration value, D(C) is the local
                          diffusion coefficient at that concentration. Should be sorted by C.
    @return: The calculated effective diffusion coefficient (D_eff) (m^2/s).
    @raises ValueError: If C0 is zero or if gradient_data has fewer than two points.
    """
    if C0 == 0:
        raise ValueError("Reference concentration (C0) cannot be zero for division.")
    if len(gradient_data) < 2:
        raise ValueError("Gradient data must contain at least two points for integration.")

    # Extract C and D(C) values
    concentrations = np.array([point[0] for point in gradient_data])
    local_diffusivities = np.array([point[1] for point in gradient_data])
    
    # --- Implementation of Numerics (Trapezoidal Rule) ---
    # The integral ∫ D(C) dC is approximated by the sum of the areas of the trapezoids
    # Area = 0.5 * (h1 + h2) * w, where h1, h2 are function values, and w is the width.
    
    # In our case, the width 'w' is (C_i+1 - C_i) and the heights are D_i and D_i+1.
    integral_sum = 0.0
    for i in range(len(concentrations) - 1):
        delta_C = concentrations[i+1] - concentrations[i]
        avg_D = (local_diffusivities[i] + local_diffusivities[i+1]) / 2.0
        integral_sum += avg_D * delta_C
        
    # Final Fickian calculation: D_eff = (1 / (2 * C0)) * integral_sum
    D_eff = (1.0 / (2.0 * C0)) * integral_sum
    
    return D_eff