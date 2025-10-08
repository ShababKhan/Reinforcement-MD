# Documentation for Reinforced Molecular Dynamics (rMD) Replication

This document serves as the central guide for the development, implementation, and validation of the rMD software based on Kolossváry and Coffey's preprint.

**Code Standard:** All Python source code must adhere strictly to **PEP 8** style guidelines.

## Introduction

*   **Goal:** To develop a modular, physics-infused generative model (Informed Autoencoder) capable of efficiently exploring the conformational space of Cereblon (CRBN) using pre-calculated Free Energy information.
*   **Reference:** Reinforce molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process. **(Cite DOI upon publication)**

## Methodology

*   **Architecture Overview:** Describe the Encoder, 3D Latent Space, Decoder, and the butterfly analogy (Fig. 2/S2).
*   **Loss Functions Detail:**
    *   **Loss 2 (Prediction Loss):** $\mathcal{L}_2$. Measures RMSD/MAE between input structure $X$ and output $\hat{X}$.
    *   **Loss 1 (Latent Loss):** $\mathcal{L}_1$. Measures correlation/distance between Latent Coordinates $L$ and Collective Variable Coordinates $CV$.
*   **Training Strategy:** Simultaneous optimization: $\text{Loss}_{\text{Total}} = \alpha \mathcal{L}_1 + \mathcal{L}_2$. (Note: $\alpha$ weight must be determined by testing to tune reconstruction vs. physics fidelity).
*   **CRBN Specifics:** Input vectors are 9696-dimensional (CRBN heavy atoms). Latent space is 3D. Activation is Swish.

## Dependencies

*   **Python Version:** X.X.X
*   **Key Libraries:**
    *   NumPy: [Version]
    *   PyTorch/TensorFlow: [Version]
    *   SciPy: [Version]
    *   MDAnalysis: [Version]
*   **External Requirements (For Full Replication):** Access to the initial MD simulation data set (10,000 frames with corresponding CV values) and the 3D Free Energy Map structure. Output post-processing requires external tools (Rosetta/Pymol - to be addressed in Phase 2).

## Tests

*   **Unit Tests:** Details of tests covering data preprocessing (superposition, flattening) and loss function calculations.
*   **Integration Tests (Convergence):** Expected final loss values from training runs: $\mathcal{L}_1 \approx 1.0, \mathcal{L}_2 \approx 1.6$.
*   **Validation Tests (Fidelity & Physics):**
    *   Fidelity Test: RMSD verification of structures generated from known LS points ($\approx 1.2 \text{ Å}$).
    *   Physics Test: Verification that structures generated from FE map regions map to known open/closed states correctly (Fig. 3/4 alignment).
*   **Path Generation Test:** Confirmation that the B-spline derived structures smoothly transition between the Open (6H0F) and Closed (6H0G) reference states.

## Glossary

*   **CV:** Collective Variable(s) (3D space).
*   **LS:** Latent Space (3D representation learned by the Encoder).
*   **FE Map:** Free Energy Map defined over the CV space.
*   **rMD:** Reinforced Molecular Dynamics.