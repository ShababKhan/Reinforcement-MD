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

#### 2. Model Architecture

The core of rMD is a fully connected (FC) autoencoder implemented in PyTorch, structured to compress the 9696-dimensional input vector into a 3-dimensional latent space ($CV_{dim}=3$).

*   **Encoder:** Gradually shrinks the dimensionality (e.g., 9696 -> 4096 -> 2048 -> 1024 -> 256 -> 3).
*   **Decoder:** Expands the dimensionality (e.g., 3 -> 256 -> 1024 -> 2048 -> 4096 -> 9696).
*   **Activation:** All hidden layers utilize the **Swish** activation function, as opposed to standard ReLU.

#### 3. Dual-Loss Optimization

The network is optimized using a weighted sum of two loss functions:

$$
\text{Total Loss} = W_1 \cdot L_1 + W_2 \cdot L_2
$$

*   **$L_1$ (Latent Loss):** Measures the correlation between the latent space coordinates ($LS$) and the input collective variables ($CV$). It uses Mean Squared Error ($\text{MSE}$) between $LS$ and $CV$: $\text{MSE}(LS, CV)$. Minimizing $L_1$ forces the latent space to map onto the physical CV space, infusing physics into the network.
*   **$L_2$ (Reconstruction Loss):** Measures the fidelity of the reconstructed structure ($\text{Pred}$) compared to the original input ($\text{Input}$). It uses Mean Absolute Error ($\text{MAE}$) between $\text{Pred}$ and $\text{Input}$: $\text{MAE}(\text{Pred}, \text{Input})$. This ensures the structural integrity of the generated protein models.

#### 4. Model Training

The `rMD_Network` is trained using the **Adam optimizer** with a learning rate of $1\times 10^{-4}$ for 10,000 epochs, using a batch size of 64 on synthetic data. The training process tracks total loss and structural RMSD on a 20% validation set to ensure non-overfitting convergence.

#### 5. Physics Infusion and Verification (Figure 3 Analogy)

The scientific validation is performed by comparing the learned Latent Space (LS) to a synthetic Free Energy (FE) map defined over the CV space. The trained `Encoder` is used to map all input frames onto the 3D LS. By assigning the $\Delta G$ value (derived from the synthetic FE map) to each point in the LS, a visual confirmation is acquired: the LS points cluster within the low-energy basins of the FE landscape, proving that the network successfully learned the intended correlation.

## Dependencies

A full list of required Python libraries and versions is provided in `requirements.txt`. Key dependencies include:

*   **PyTorch (`torch`):** For building, training, and optimizing the neural network models.
*   **NumPy (`numpy`):** For efficient manipulation of high-dimensional numerical data, synthetic data generation, and array operations.
*   **Matplotlib (`matplotlib`):** Required for visualizations of loss curves, the 3D free-energy map, and the trained latent space.

## Tests

The project includes unit and integration tests to guarantee code integrity and scientific accuracy.

1.  **Data Integrity Tests:** Confirmed the correct dimensions and removal of translational invariance during preprocessing.
2.  **Model and Loss Tests:** Confirmed correct dimensionality of the autoencoder and proper arithmetic functionality of the Dual-Loss implementation.
3.  **Integration Tests (QA Checkpoint 3):** Verified training convergence and successful demonstration of the LS-FE correlation via visualization.
