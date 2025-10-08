import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rmd_model import rMD_Network, CV_DIM

# --- Configuration ---
# Match the range of CVs defined in data_utils (approx -1 to 1)
GRID_MIN = -1.5
GRID_MAX = 1.5
GRID_RESOLUTION = 30  # Determines the number of voxels (30^3 = 27000 points in FE map)
CV_DIM = 3

# --- U5.1: FE Map Data Structure ---
def generate_synthetic_fe_map() -> np.ndarray:
    """
    Generates a synthetic 3D Free Energy (FE) map defined over the CV space.
    
    The synthetic map is designed to have a few low-energy basins (valleys)
    and high-energy regions (barriers/peaks), mimicking a realistic FE landscape.
    This FE map is the 'physical ground truth' that the rMD model's latent
    space should align with.
    
    @return: A 3D NumPy array representing the relative free energy (Delta G).
    """
    print(f"Generating synthetic FE map with resolution {GRID_RESOLUTION}^3...")
    
    # Create grid axes
    x_range = np.linspace(GRID_MIN, GRID_MAX, GRID_RESOLUTION)
    y_range = np.linspace(GRID_MIN, GRID_MAX, GRID_RESOLUTION)
    z_range = np.linspace(GRID_MIN, GRID_MAX, GRID_RESOLUTION)
    
    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    # Define a complex landscape function (e.g., sum of Gaussian potentials)
    # Simple double-well potential analogy:
    # Basin 1 (Open): near (-0.8, 0, 0)
    # Basin 2 (Closed): near (0.8, 0, 0)
    
    # Potential 1 (Basin around open state)
    pot1 = 5 * np.exp(-10 * ((X + 0.8)**2 + Y**2 + Z**2))
    # Potential 2 (Basin around closed state)
    pot2 = 5 * np.exp(-10 * ((X - 0.8)**2 + Y**2 + Z**2))
    # Peak (Transition barrier)
    peak = 7 * np.exp(-1 * (X**2 + Y**2 * 0.5 + Z**2 * 0.5))
    
    # Total Free Energy (Delta G) - inverted for visualization: Minima are low G
    # We use (Potential Sum - Max Potential Sum) to normalize such that
    # the maximum energy difference is max_FE_value (e.g., 15 kcal/mol)
    RAW_ENERGY = pot1 + pot2 - peak
    
    # Normalize to 0 (min) to 15 (max) kcal/mol (standard FE range)
    MAX_FE_VAL = 15.0
    G = MAX_FE_VAL - (RAW_ENERGY - np.min(RAW_ENERGY))
    G = np.clip(G, 0, MAX_FE_VAL)
    
    return G

# --- U6.2: Color Mapping Utility ---
def get_fe_value(cv_points: np.ndarray, fe_map: np.ndarray) -> np.ndarray:
    """
    Snaps each 3D CV point to the nearest grid point on the FE map 
    and returns the corresponding Free Energy (G) value.
    
    @param cv_points: Array of 3D points (N, 3) from the latent space/CVs.
    @param fe_map: The generated FE map grid (GRID_RESOLUTION, GRID_RESOLUTION, GRID_RESOLUTION).
    @return: Array of FE values corresponding to each input CV point (N,).
    """
    # Define grid range and size for mapping
    grid_size = fe_map.shape[0]
    
    # Normalize CV coordinates to grid indices [0, GRID_RESOLUTION - 1]
    # Indices = round((CV - GRID_MIN) / (GRID_MAX - GRID_MIN) * (grid_size - 1))
    scale_factor = (grid_size - 1) / (GRID_MAX - GRID_MIN)
    indices_float = (cv_points - GRID_MIN) * scale_factor
    
    # Clip indices to ensure they are within the bounds [0, grid_size - 1]
    indices_clipped = np.clip(np.round(indices_float), 0, grid_size - 1).astype(int)
    
    # Extract FE values using the mapped indices
    fe_values = fe_map[indices_clipped[:, 0], indices_clipped[:, 1], indices_clipped[:, 2]]
    
    return fe_values

# --- Main Analysis and Visualization ---
def run_analysis(model_path='rmd_best_model.pt'):
    """
    Loads the trained model, performs latent space compression, and generates 
    the comparative visualizations (Fig. 3 analogy).
    """
    device = torch.device("cpu")
    
    # 1. Load Data
    try:
        coords_flat = np.load('synthetic_coords.npy')
        cvs = np.load('synthetic_cvs.npy')
    except FileNotFoundError:
        print("Error: Synthetic data not found. Ensure synthetic_coords.npy and synthetic_cvs.npy exist.")
        return

    # 2. Load Model (T6.1 preparation)
    model = rMD_Network()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"Error: Trained model state not found at {model_path}. Please run train.py first.")
        return
        
    # 3. Generate Latent Space coordinates (LS points) (T6.1)
    print("Generating Latent Space coordinates from trained model...")
    with torch.no_grad():
        coords_tensor = torch.tensor(coords_flat, dtype=torch.float32).to(device)
        ls_coords, _ = model(coords_tensor)
        ls_coords_np = ls_coords.cpu().numpy()

    # 4. Generate Free Energy Map (T5.1)
    fe_map = generate_synthetic_fe_map()
    
    # 5. Map FE values to the Latent Space points (T6.2)
    # The LS points should theoretically align with the FE map based on the training
    ls_fe_values = get_fe_value(ls_coords_np, fe_map)
    
    # 6. Visualization (T5.2 & T6.1) - Analogy to Figure 3
    print("Generating comparative 3D scatter plots (Analogy to Figure 3)...")
    fig = plt.figure(figsize=(15, 7))
    
    # --- Plot 1: Latent Space colored by FE (Left side of Fig. 3) ---
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Scatter plot, color based on the mapped FE values
    # Use 'viridis_r' colormap (green-yellow-red gradient for low-high FE)
    sc1 = ax1.scatter(ls_coords_np[:, 0], ls_coords_np[:, 1], ls_coords_np[:, 2], 
                      c=ls_fe_values, cmap='viridis_r', s=10, alpha=0.6, vmin=0, vmax=15)
    
    # Set plot limits based on the synthetic data range
    ax1.set_xlim([GRID_MIN, GRID_MAX])
    ax1.set_ylim([GRID_MIN, GRID_MAX])
    ax1.set_zlim([GRID_MIN, GRID_MAX])
    
    ax1.set_title("Trained Latent Space (Colored by Free Energy)")
    ax1.set_xlabel("LS Dim 1")
    ax1.set_ylabel("LS Dim 2")
    ax1.set_zlabel("LS Dim 3")
    fig.colorbar(sc1, ax=ax1, label='Relative Free Energy (kcal/mol)')

    # --- Plot 2: Synthetic Free Energy Map Density (Right side of Fig. 3) ---
    ax2 = fig.add_subplot(122, projection='3d')
    
    # To visualize the 3D FE map grid, we display points only where FE is non-zero
    # to highlight the structure, similar to a density plot.
    X, Y, Z = np.meshgrid(
        np.linspace(GRID_MIN, GRID_MAX, GRID_RESOLUTION), 
        np.linspace(GRID_MIN, GRID_MAX, GRID_RESOLUTION), 
        np.linspace(GRID_MIN, GRID_MAX, GRID_RESOLUTION), 
        indexing='ij'
    )
    
    # Flatten the grid coordinates and FE values
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    FE_flat = fe_map.flatten()
    
    # Filter points to only show relevant regions (e.g., below 10 kcal/mol)
    mask = FE_flat < 10

    sc2 = ax2.scatter(X_flat[mask], Y_flat[mask], Z_flat[mask],
                      c=FE_flat[mask], cmap='viridis_r', s=5, alpha=0.3, vmin=0, vmax=15)

    ax2.set_xlim([GRID_MIN, GRID_MAX])
    ax2.set_ylim([GRID_MIN, GRID_MAX])
    ax2.set_zlim([GRID_MIN, GRID_MAX])
    
    ax2.set_title("Synthetic Free Energy Map (CV Space)")
    ax2.set_xlabel("CV Dim 1")
    ax2.set_ylabel("CV Dim 2")
    ax2.set_zlabel("CV Dim 3")
    fig.colorbar(sc2, ax=ax2, label='Relative Free Energy (kcal/mol)')

    plt.tight_layout()
    plt.show()
    print("Analysis complete. Comparative visualization displayed.")


if __name__ == '__main__':
    run_analysis()