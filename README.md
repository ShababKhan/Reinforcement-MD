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

#### 2. Model Architecture: The Informed Autoencoder (T2.1 - T2.3)

The core architecture is an Autoencoder, implemented using PyTorch, composed of an Encoder and a Decoder network connected via a 3-dimensional Latent Space (LS).

1. **Encoder:** A series of fully connected (FC) layers gradually compresses the 9696-dimensional input vector down to the 3-dimensional latent space vector.
2. **Decoder:** A series of FC layers expands the 3-dimensional LS vector back to the 9696-dimensional output vector (the reconstructed coordinates).
3. **Activation:** All hidden layers in both the Encoder and Decoder utilize the **Swish** activation function ($\text{x} \sigma(\text{x})$), as specified in the paper's network design implicitly derived from the context of VAEs and deep learning in structural biology.

**Scientific Link:** The 3-dimensional Latent Space is the key physical concept, designed to directly correspond to the 3-dimensional Collective Variable (CV) space where the Free Energy (FE) map is defined.

#### 3. Dual-Loss Optimization (T3.1 - T3.3)

The network is trained by simultaneously minimizing two loss functions, injecting physics into the optimization process. This process uses a weighted sum optimization: $L_{\text{Total}} = W_1 \cdot L_1 + W_2 \cdot L_2$.

1. **$L_1$ (Latent Loss):** Measures the discrepancy between the Latent Space coordinates produced by the Encoder and the ground truth Collective Variable ($\text{CV}$) targets. This is calculated using Mean Squared Error (MSE) on the 3D vectors. Minimizing $L_1$ forces the network to learn the physics-based map (CV $\rightarrow$ LS).
2. **$L_2$ (Reconstruction Loss):** Measures the fidelity of the reconstructed coordinates against the original input coordinates, often reported as RMSD. This is calculated using Mean Absolute Error (MAE) (L1 Loss) on the high-dimensional vectors, which is a robust metric for structural dissimilarity. Minimizing $L_2$ ensures the network performs accurate structure reconstruction.

**Scientific Link:** The dual-loss approach explicitly links the abstract machine learning latent space ($L_1$) to the scientifically meaningful collective variable space, while maintaining structural correctness ($L_2$).

## Dependencies

A full list of required Python libraries and versions is provided in `requirements.txt`. Key dependencies include:

*   **PyTorch (`torch`):** For building, training, and optimizing the neural network models.
*   **NumPy (`numpy`):** For efficient manipulation of high-dimensional numerical data, synthetic data generation, and array operations.
*   **Matplotlib (`matplotlib`):** Required for visualizations of loss curves, the 3D free-energy map, and the trained latent space (Sprint 2).

## Tests

This section will document the verification process:

1.  **Unit Tests:** Verification of data integrity, loss function correctness, and module functionality.
2.  **Integration Tests:** Verification of the full training pipeline using synthetic data.
3.  **Scientific Validation:** Confirmation that the trained latent space visually correlates with the synthetic Free Energy map (as measured by the coloring utility).

Initial unit tests for data preprocessing (`tests/test_data_utils.py`) have confirmed:
*   Data dimensions meet the expected `(N_frames, 9696)` for coordinates.
*   The `superpose_and_flatten` utility correctly eliminates global translational degrees of freedom by centering the input structures before training.
