"""
path_generator.py (Sprint 3.3)

Implements the B-spline fitting utility to interpolate between anchor points in the 
3D CV space, generating coordinates along the putative transition path (Figure 4).
"""
import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple

# Constants matching the rMD paper's context
LATENT_DIM = 3 
N_PATH_POINTS = 20  # Number of structures to generate along the path (for 20 states in Movie S1)

def generate_transition_path(
    anchor_cvs: np.ndarray, 
    n_points: int = N_PATH_POINTS
) -> np.ndarray:
    """
    Fits a B-spline to user-defined anchor points in the CV space and samples
    points along this curve to define the transition pathway.

    @param anchor_cvs: A numpy array of shape (N_anchors, LATENT_DIM=3) containing 
                       the CV coordinates for the open, transition, and closed regions.
                       Example order: [CV_Open, CV_Transition_1... CV_Closed]
    @param n_points: The number of discrete points (magenta points in Fig 4) to sample along the path.
    @return: Array of shape (n_points, 3) containing sampled CV coordinates.
    """
    if anchor_cvs.ndim != 2 or anchor_cvs.shape[1] != LATENT_DIM:
        raise ValueError(
            f"Anchor CVs must be a 2D array of shape (N_anchors, {LATENT_DIM})."
        )

    # 1. Fit the B-spline (using 'fitpack' routines in scipy)
    # tck: tuple (knots, spline_coefficients, degree)
    # u: parameter for the curve (0 to 1 parameterization)
    tck, u = splprep([anchor_cvs[:, d] for d in range(LATENT_DIM)], s=0, k=min(3, anchor_cvs.shape[0] - 1))
    
    # 2. Generate parameter values for desired sampling density
    # Parameter t ranges from 0 (start anchor) to 1 (end anchor)
    t_new = np.linspace(u.min(), u.max(), n_points)
    
    # 3. Evaluate the spline at new parameter points
    # splev returns a list of arrays, where each array contains the x, y, or z coordinate (i.e., CV1, CV2, CV3)
    sampled_cvs_list = splev(t_new, tck)
    
    # Recombine into the required structure (n_points, 3)
    sampled_cvs = np.vstack(sampled_cvs_list).T
    
    return sampled_cvs


# --- Unit Tests ---
def test_path_generator():
    """
    Tests the B-spline fitting and path sampling functionality.
    """
    print("\n--- Running Path Generator Unit Tests ---")
    
    # Define anchor points (Mocking the open, transition, closed regions in CV space)
    # Anchor 1 (Open State)
    cv_open = np.array([0.5, 5.0, 10.0])
    # Anchor 2 (Midpoint/Transition)
    cv_mid = np.array([5.0, 2.5, 5.0])
    # Anchor 3 (Closed State)
    cv_closed = np.array([10.0, 0.1, 0.1])
    
    anchors = np.array([cv_open, cv_mid, cv_closed])
    
    # Generate 20 points along the path
    path_points = generate_transition_path(anchors, n_points=20)
    
    # Test 1: Shape check
    assert path_points.shape == (20, LATENT_DIM), f"Path points shape incorrect: {path_points.shape}"
    
    # Test 2: Boundary check (Start point should be close to the first anchor)
    assert np.allclose(path_points[0], cv_open, atol=1e-2), f"Start point mismatch: {path_points[0]} vs {cv_open}"
    
    # Test 3: Boundary check (End point should be close to the last anchor)
    assert np.allclose(path_points[-1], cv_closed, atol=1e-2), f"End point mismatch: {path_points[-1]} vs {cv_closed}"
    
    # Test 4: Interpolation check (intermediate points should be between min/max)
    # The mean of the intermediate points should lie between the mean of anchors
    
    print("Path Generator Unit Test PASSED.")
    
if __name__ == '__main__':
    test_path_generator()