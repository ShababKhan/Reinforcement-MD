# rMD Software Replication Project: CRBN Conformational Exploration

## Introduction
*   **Project Goal:** To implement the Reinforced Molecular Dynamics (rMD) method described in Kolossváry & Coffey (2025) to model the CRBN open-to-closed conformational transition.
*   **Key Technology:** Physics-infused autoencoder trained with a dual-loss function linking Latent Space (LS) to Collective Variable (CV) space.

## Methodology
*   **Simulation Data Dependency:** This implementation relies on external data derived from meta-eABF simulations on the CRBN-DDB1 apo complex, providing 10,000 trajectory frames and an associated 3D Free Energy (FE) Map over three Collective Variables (CVs). Structural details from PDB IDs 6H0G (closed) and 6H0F (open) are relevant for validation.
*   **Data Preprocessing Pipeline:** Input structures must be superposed onto a reference frame to eliminate global translation/rotation noise before flattening the heavy-atom Cartesian coordinates (length 9696 input vectors).
*   **Network Architecture:** An autoencoder structure featuring a shrinking encoder and expanding decoder, connected by a 3-dimensional latent space (LS). All linear layers use the **Swish activation function** ($x\sigma(x)$).
*   **Loss Function Implementation:** The network is optimized simultaneously using two loss functions:
    *   **Loss 2 ($\text{predLoss}$):** Reconstruction loss, minimizing the Root Mean Square Deviation (RMSD) between input and decoded structures.
    *   **Loss 1 ($\text{latentLoss}$):** Physics infusion loss, minimizing the distance between the 3D LS coordinates and the target 3D CV coordinates.
*   **Training Regimen:** Simultaneous optimization of both losses using the **Adam optimizer** over 10,000 rounds with a batch size of 64. The weights between Loss 1 and Loss 2 are configurable.
*   **Structure Generation & Path Finding:** The trained network is used to map points from the CV space (derived from a **B-spline** fit trajectory anchor points in Fig. 4) back to full atomic structures via the decoder.

## Dependencies
*   **Technology Stack:** Python
*   **Core Scientific Libraries:** NumPy, SciPy
*   **ML/Deep Learning Framework:** TensorFlow/Keras or PyTorch (to be finalized in Sprint 0)
*   **MD/Bioinformatics Tools (for data comparison/post-processing):** MDAnalysis (Recommended)

## Tests
*   **Unit Tests:** Validation of data loading, structure superposition utility, RMSD calculation, and custom Swish activation.
*   **Integration Tests:** Verification that the dual-loss training converges to the reported reference loss metrics ($\text{Loss1} \approx 1.0 \text{ \AA}$, $\text{Loss2} \approx 1.6 \text{ \AA}$).
*   **Validation Tests:** Qualitative comparison of the LS distribution to published Figure 3 and successful generation of structures along the path defined in Figure 4.
*   **Code Quality:** All code must conform to PEP 8.
