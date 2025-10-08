# Reinforced Molecular Dynamics (rMD) Recreation Project

## Introduction

This project aims to digitally replicate the Reinforced Molecular Dynamics (rMD) model developed in the paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" (Kolossváry & Coffey, bioRxiv 2025). The core objective is to create a functional, scientifically valid software agent that can reproduce the key findings of the paper, specifically the physics-infused autoencoder used to model the open-to-closed state transition of the CRBN protein.

## Methodology

The rMD approach centers around an informed autoencoder trained with a dual-loss function:

1.  **Loss 2 (Prediction Loss):** Minimizes the structural reconstruction error (RMSD ≈ 1.6 Å) between the input protein coordinates and the decoded output coordinates.
2.  **Loss 1 (Latent Loss):** Forces the 3-dimensional Latent Space (LS) coordinates to align with the 3-dimensional Collective Variables (CVs) derived from the MD simulation trajectory (RMSD ≈ 1.0 Å).

This parallel optimization infuses the latent space with physical meaning, allowing for targeted structure generation based on coordinates selected from a calculated Free Energy map.

The development process will follow the Project Blueprint laid out in the management communication, beginning with data preparation and the development of core components.

## Dependencies

The entire solution will be implemented in Python using the following key libraries:

| Dependency | Purpose |
| :--- | :--- |
| `PyTorch` | Core deep learning framework for the autoencoder and custom loss functions. |
| `NumPy` | Numerical operations and array handling. |
| `MDAnalysis` | **Required** for loading molecular data, calculating Center of Mass (COM), and performing structure superposition/alignment. |
| `SciPy` | B-Spline interpolation for path generation. |
| `tqdm` | Training progress reporting. |

## Tests

This section will be populated by the QA Engineer following the execution and validation of the code against the scientific benchmarks identified in the source paper.

### Sprint 1 Validation Benchmarks:

*   **CV Calculation Logic:** Verify that the logic accurately defines the Collective Variables based on the COM distances of the four specified domains (CRBN-NTD, CRBN-CTD, CRBN-HBD, DDB1-BPC).
*   **Data Structure:** Confirm input vector length is 9696 (heavy atoms of CRBN only).
*   **Data Split:** Confirm 8000 training / 2000 validation split.

### Sprint 2 Validation Benchmarks:

*   **Model Performance (Losses):** Final Loss1 $\approx 1.0 \text{\AA}$ and Loss2 $\approx 1.6 \text{\AA}$.
*   **Physics Infusion:** Demonstrate a strong correlation between the 3D Latent Space coordinates and the 3D CV coordinates for the validation set.
*   **Path Generation:** Verify the B-spline interpolation accurately defines a continuous path in CV space as a precursor to structure generation.

---