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

#### 2. Model Architecture and Dual-Loss Optimization

The core model, implemented in PyTorch, is an autoencoder utilizing a dual-loss function for physics infusion.

*   **Network:** The `rMD_Network` consists of a fully connected `Encoder` and `Decoder` structure with gradually decreasing/increasing hidden layers. The architecture is constrained by a 3-dimensional latent space.
*   **Activation:** The non-linear activation function used throughout the hidden layers is the **Swish** function, consistent across the Encoder and Decoder (except for the final layers).
*   **Dual-Loss Function:** The network is optimized to minimize a weighted sum of two loss components:
    *   $L_1$ (Latent Loss / `latentLoss`): Measures the Mean Squared Error (or RMSD) between the 3D Latent Space coordinates and the 3D Collective Variables (CVs). This forces the Latent Space to acquire physical meaning.
    *   $L_2$ (Reconstruction Loss / `predLoss`): Measures the Mean Absolute Error (MAE) between the input Cartesian coordinates and the final reconstructed coordinates. This ensures structural fidelity.

#### 3. Model Training

The `rMD_Network` is trained using the Adam optimizer with a learning rate of $1\times 10^{-4}$ for 10,000 steps, using a batch size of 64.

1.  **Data Split:** The total synthetic dataset (10,000 frames) is split into 80% for training ($N=8000$) and 20% for validation ($N=2000$).
2.  **Optimization:** The training minimizes the `DualLoss` function, which is the weighted sum of $L_1$ and $L_2$.
3.  **Tracking:** Validation is performed every 100 epochs, tracking overall validation loss and the reconstruction RMSD (root mean square deviation) as the primary measure of structural prediction quality. The model with the lowest validation loss is saved.

#### 4. Physics Infusion and Verification (Figure 3 Analogy)

The core scientific validation of the rMD method lies in the correlation between the learned Latent Space (LS) and the known Free Energy (FE) map defined over the Collective Variable (CV) space.

1.  **Synthetic FE Map:** A 3D grid is constructed in the CV space (matching the 3D LS size) storing synthetic $\Delta G$ values, mimicking a realistic double-well potential with a transition barrier.
2.  **LS Extraction:** The trained `Encoder` is run on the entire trajectory data to extract the 3D LS coordinates for all frames.
3.  **FE Mapping:** Each extracted LS point is 'snapped' to the closest grid point in the synthetic FE map. This assigns a physical meaning (a $\Delta G$ value) to the otherwise arbitrary LS coordinate.
4.  **Verification:** The LS points are visualized, colored by their assigned $\Delta G$ value. A successful training run demonstrates high fidelity, showing that points that map to low-energy regions (basins) in the FE map are concentrated in distinct clusters of the LS plot. This structural similarity confirms that the dual-loss function successfully forced the latent space to inherit the physical context of the CV space.

---

## Dependencies

A full list of required Python libraries and versions is provided in `requirements.txt`. Key dependencies include:

*   **PyTorch (`torch`):** For building, training, and optimizing the neural network models.
*   **NumPy (`numpy`):** For efficient manipulation of high-dimensional numerical data, synthetic data generation, and array operations.
*   **Matplotlib (`matplotlib`):** Required for visualizations of loss curves, the 3D free-energy map, and the trained latent space.

## Tests

The project followed a rigorous verification protocol across two Sprints:

*   **QA Checkpoint 1 (Data & Preprocessing):** Verified data dimensions, implementation of the `superpose_and_flatten` utility, and confirmation that the synthetic data structure enforces a correlation between coordinates and CVs.
*   **QA Checkpoint 2 (Model & Loss):** Verified the correct dimensionality of the Encoder/Decoder, the inclusion of the Swish activation function, and the correct arithmetic calculation of the non-negative `DualLoss` function.
*   **QA Checkpoint 3 (Integration & Science):** Verified the end-to-end pipeline, confirming model convergence during training and scientifically validating the result through the comparative 3D visualization. The latent space geometry successfully mirrored the synthetic Free Energy landscape, confirming the core principle of **Reinforced Molecular Dynamics**.