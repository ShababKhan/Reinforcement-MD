# Reinforced Molecular Dynamics (rMD) Project Documentation

This document serves as the central repository for all documentation related to the recreation of the Reinforced Molecular Dynamics (rMD) inference model, as described by Kolossváry and Coffey.

---

## 1. Introduction

*   **Project Goal:** To implement the Reinforced Molecular Dynamics ($\text{rMD}$) generative model that combines Molecular Dynamics (MD) trajectory data with a Free Energy (FE) map to inform an autoencoder for efficient exploration of protein conformational space.
*   **Target System:** Cereblon (CRBN) open-to-closed conformational transition.
*   **Code Standard:** All Python code **must** adhere to PEP 8 standards.

## 2. Methodology

This section will be populated during the implementation phase, detailing the specifics for each component:

*   **MD Simulation Prerequisite:** Details on meta-eABF setup, CV definitions (Fig. S1), FF used, and resulting FE map generation. (To be filled by MD/Data Engineer Agent).
*   **Autoencoder Architecture:** Precise layer dimensions, number of parameters, and details on the encoder/decoder structure (inspired by Ref [9], detailed in Fig. S2). (To be filled by ML Engineer Agent).
*   **Loss Functions:** Detailed mathematical definition and weighting scheme for $\text{Loss}_1$ (Latent Loss) and $\text{Loss}_2$ (Reconstruction Loss).
*   **Training Procedure:** Hyperparameters (Optimizer: Adam/Adams, Learning Rate schedule, Epochs, Batch Size, $\text{Loss}_1 / \text{Loss}_2$ weighting factor).

## 3. Dependencies

*   **Technology Stack:** Python (Primary Implementation Language).
*   **Core Libraries** (To be confirmed and installed):
    *   Data Handling: NumPy, Pandas
    *   Machine Learning Framework: TensorFlow or PyTorch (to be decided in Sprint 1 Planning)
    *   MD Trajectory/Structure Handling: MDAnalysis, Biopython (or similar).
*   **External Tool Dependencies (For Validation/Post-processing):**
    *   Wolfram Mathematica (for initial model definition/verification).
    *   Rosetta Suite (for structure relaxation/post-processing).

## 4. Tests

This section details the quantitative and qualitative tests required to confirm the successful recreation of the reported results.

*   **Quantitative Tests (Acceptance Criteria Mapping):**
    *   Training convergence: $\text{Loss}_2$ (Heavy-atom RMSD) must reach $\approx 1.6$ Å.
    *   Physics Injection fidelity: $\text{Loss}_1$ must be minimized, confirmed by visual overlap between LS and CV space density plots (Fig. 3).
*   **Qualitative Tests:**
    *   Structure Quality: Post-processed structures must be visually and sterically plausible compared to PDB data.
    *   Path Validation: The generated transition path (Fig. 4) must qualitatively match the topology of the FE map (narrow path between basins).
*   **Deliverable Checks:**
    *   Movie S1 and Data S1 (Pymol file) must be generated from the final predicted path points.

---
*Last Updated: [Date]*