# Project Blueprint: Reinforced Molecular Dynamics (rMD) Recreation

## Introduction
This document outlines the plan to replicate the Reinforced Molecular Dynamics (rMD) methodology described in Kolossváry & Coffey (2025). The goal is to implement a dual-loss autoencoder network that integrates a pre-computed Free Energy (FE) map, derived from Molecular Dynamics (MD) simulations of the CRBN conformational change, into the latent space.

## Methodology
The core scientific methodology relies on linking the autoencoder's latent space to the collective variable (CV) space defined by the simulation.

**Key Scientific Concepts to Implement:**
1.  **MD Simulation Data:** Data representing the CRBN open/closed transition (input coordinates from 10,000 heavy-atom structures).
2.  **Collective Variables (CVs):** Three CVs defined by the relative distances (COMs) of CRBN domains (CRBN-NTD, CTD, HBD) and DDB1-BPC (Figure S1).
3.  **Autoencoder Structure:** Encoder/Decoder using fully connected layers with **Swish** activation.
4.  **Dual Loss Function:**
    *   **Loss2 (Reconstruction):** Average RMSD between input and output structures.
    *   **Loss1 (Latent Physics Infusion):** RMSD between Latent Space coordinates and target CV coordinates.
5.  **Application:** Generating structures along a low-energy path defined by a B-spline in CV space (Figure 4).

**Code Quality Mandate:** All Python code must adhere strictly to **PEP 8 standards**.

## Dependencies
The software components require the following technology stack:
*   **Technology:** Python 3.x
*   **Primary ML Library:** PyTorch (or TensorFlow)
*   **Structural/Data Handling:** NumPy, SciPy, MDAnalysis
*   **Wolfram:** Implementation details (Layer structure, specific loss weighting) must be cross-referenced with the paper/Supplementary Material.

## Tests
All components must include unit tests. Key milestones for validation:
1.  **Data Preprocessing Test:** Verify that CV calculation for known structures (PDB 6H0G, 6H0F) yields stable, correct values.
2.  **Model Fit Test:** Verify that the network achieves $\text{Loss}_2 \approx 1.6 \text{ Å}$ and $\text{Loss}_1 \approx 1.0 \text{ Å}$ on the training/validation sets, reflecting the paper's results (T2.5).
3.  **Physics Correlation Test:** Verify that sampling structures from the LS and plotting them against the FE map resembles Figure 3.
4.  **Transition Test:** Verify that structures generated along the B-spline path successfully navigate between the open and closed states (as seen in Movie S1).
