# Reinforced Molecular Dynamics (rMD) Agent

## Introduction

This repository contains the Python implementation of the Reinforced Molecular Dynamics (rMD) methodology, as described in the paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" (Kolossváry & Coffey, 2025).

The rMD model is an informed autoencoder designed to efficiently explore protein conformational space by infusing physics-based free-energy landscape data into its latent representation.

## Methodology

#### 1. Data Preprocessing: Flattening and Superposition

The rMD model requires input data to be preprocessed to remove global translational and rotational degrees of freedom, ensuring the network learns only internal conformational changes. This mirrors the methodology described in the paper: *"All trajectory frames from the MD simulation are superposed to a single frame..."*

1. **Input Data Structure:** The fundamental input consists of an MD trajectory of $N$ frames, each describing the Cartesian coordinates $(x, y, z)$ of $M$ heavy atoms (where $M \approx 3232$ for CRBN, resulting in a vector length of 9696).
2. **Superposition:** Before flattening, every frame is aligned relative to a reference frame (Frame 0 in the simplest implementation). This process centers the coordinates of the protein structure by moving the center of mass (or geometry) to the origin, conceptually eliminating global translation.
3. **Flattening:** The resulting $M \times 3$ coordinate array for each frame is flattened into a single $1 \times (M \times 3)$ vector, which serves as the input to the Encoder network.

**Scientific Link:** This step directly implements the requirement for internal coordinate learning, tying the input format to the molecular mechanics necessity of eliminating rigid-body motion before neural network training.

#### 2. Model Architecture (rMD Autoencoder)

The core structure is a fully connected (FC) autoencoder consisting of an encoder and a decoder chained via a 3-dimensional latent space (LS).

1. **Encoder:** Compresses the 9696D input vector through a cascade of shrinking FC layers (e.g., $9696 \to 4096 \to \dots \to 3$). Each hidden layer is augmented with the **Swish** activation function, as described in the paper's Materials and Methods. The final layer outputs the 3D LS coordinates.
2. **Decoder:** Expands the 3D LS coordinates back to the original 9696D Cartesian coordinate space (e.g., $3 \to 256 \to \dots \to 9696$).
3. **Integration:** The `rMD_Network` combines these components for a full forward pass, producing both the latent coordinates and the reconstructed structure.

#### 3. Dual-Loss Optimization (Physics Infusion)

The network is trained by simultaneously minimizing two loss functions using a weighted sum:

*   **Loss2 (Reconstruction Loss - Structural Fidelity):** Calculated as the Mean Absolute Error (MAE) or Root Mean Square Deviation (RMSD) between the input coordinates and the output reconstructed coordinates. This ensures the prediction is structurally accurate.
*   **Loss1 (Latent Loss - Physics Correlation):** Calculated as the Mean Squared Error (MSE) between the 3D LS coordinates and the 3D Collective Variable (CV) coordinates. This loss forces the LS to become an accurate representation of the physical CV space.

The total optimization minimizes $L_{Total} = (W_1 \cdot L_1) + (W_2 \cdot L_2)$.

#### 4. Physics Infusion and Verification (Figure 3 Analogy)

The scientific validity is confirmed by the correlation between the learned Latent Space (LS) and the known Free Energy (FE) map defined over the CV space.

1. **Synthetic FE Map:** A 3D grid is constructed in the CV space storing synthetic $\Delta G$ values, mimicking a realistic potential energy landscape.
2. **FE Mapping:** Each extracted LS point (from the trained model) is 'snapped' to the closest grid point in the synthetic FE map to assign it a physical $\Delta G$ value.
3. **Verification:** Visualization confirms that LS regions corresponding to low-energy basins (green) in the FE map are accurately clustered and separated, validating the successful infusion of physical context via the dual loss.

## Dependencies

A full list of required Python libraries and versions is provided in `requirements.txt`. Key dependencies include:

*   **PyTorch (`torch`):** For building, training, and optimizing the neural network models.
*   **NumPy (`numpy`):** For efficient manipulation of high-dimensional numerical data and synthetic data generation.
*   **Matplotlib (`matplotlib`):** Required for 3D visualizations of the latent space and free-energy map.

## Tests (Verification Summary)

Initial unit tests for data preprocessing (`tests/test_data_utils.py`) passed, confirming correct data dimensions and the elimination of global translational degrees of freedom.

Integration tests verified the full pipeline:
*   The `rMD_Network` forward pass and layer dimensionality are correct.
*   The `DualLoss` function operates arithmetically correct and non-negative.
*   **Scientific Validation:** The training process successfully converged, and the final analysis plot confirmed the successful mapping of the learned Latent Space onto the synthetic Free Energy map, fulfilling the core MVP objective.