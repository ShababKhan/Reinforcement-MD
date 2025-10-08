# Reinforced Molecular Dynamics (rMD) Master Documentation

## Introduction

This project aims to recreate the Reinforced Molecular Dynamics (rMD) framework described in the paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" (Kolossváry & Coffey, bioRxiv 2025). The core goal is to implement a dual-loss autoencoder (rMD) capable of mapping a pre-computed 3-dimensional Collective Variable (CV) space to the network's latent space.

This capability allows for the targeted generation of biologically relevant protein conformations, specifically the open-to-closed state transition of the CRBN protein (PDB IDs 6H0F $\to$ 6H0G).

## Agile Project Plan (Sprints)

| Sprint | Goal | Key Tasks | Acceptance Criteria (AC) | Duration | Team Focus |
|:---|:---|:---|:---|:---|:---|
| **0 (Setup)** | Initialization & Data Skeleton | **T0.1 (PM):** Initialize repository, structure, `README`, and documentation. **T0.2 (PM):** Define and document dependencies (`requirements.txt`). **T0.3 (Dev):** Create mock data arrays (`coords.npy`, `cvs.npy`) simulating 10,000 frames (9696 coords, 3 CVs). | **AC 0.1:** Project structure and basic files are committed. **AC 0.2:** Mock data created with correct shape (10000x9696 and 10000x3). | 1 Day | PM/Dev |
| **1 (Data & Loss)** | Data Pipeline and Physics-Infused Loss Functions | **T1.1 (Dev):** Implement `rmd.data_loader` (load, split, batch, GPU/Tensor conversion). **T1.2 (Dev):** Implement the dual-loss function: $\text{Loss}_2$ (Reconstruction, RMSD/MAE on 9696D) and $\text{Loss}_1$ (Latent, RMSD/MAE on 3D CVs). **T1.3 (QA):** Write unit tests for data loader and loss functions. | **AC 1.1:** Data loader outputs correct tensor shapes. **AC 1.2:** Loss functions are mathematically correct and stable on dummy inputs. **AC 1.3:** All S1 unit tests Pass. | 3 Days | Dev/QA |
| **2 (Model Core)** | Autoencoder Architecture and Training Loop | **T2.1 (Dev):** Implement the FC-NN Autoencoder (9696 $\to$ 3 $\to$ 9696) using **Swish** activation. **T2.2 (Dev):** Implement the full training script (Adams, 10,000 steps, batch 64, weighted $\text{Loss}_1 + \text{Loss}_2$). **T2.3 (QA):** Write integration test for training loop stability and preliminary loss decrease. | **AC 2.1:** Model forward/backward passes complete successfully. **AC 2.2:** Training script saves a checkpoint. **AC 2.3:** Integration test confirms system stability. | 4 Days | Dev/QA |
| **3 (Validation & Generation)**| Validation to Paper's Results & Structure Generation | **T3.1 (Dev):** Train model to full convergence. **T3.2 (Dev):** Implement `rmd.generate` script to use SciPy interpolation on a simulated CV path $\to$ Decoder $\to$ structure generation. **T3.3 (QA/Critical):** **Scientific Validation Test 1:** Verify converged losses against paper: $\text{Loss}_1 \approx 1.0 \text{Å}$ and $\text{Loss}_2 \approx 1.6 \text{Å}$ ($\pm 10\%$). **T3.4 (QA/Critical):** **Scientific Validation Test 2:** Verify generated structures are chemically sound and represent a transition path (e.g., visual correlation to Fig. 3). | **AC 3.1 (MVP):** Final loss values reported match expectations. **AC 3.2:** The generation script produces a valid molecular trajectory output. **AC 3.3:** The project is declared Scientifically Validated by QA. | 5 Days | Dev/QA |

## Methodology

### Informed Autoencoder Design
The autoencoder is a fully connected (FC) neural network designed to compress 9696 Cartesian coordinates into a 3-dimensional latent space.
*   **Input Dimension:** 9696 (All heavy atoms of CRBN, pre-superposed).
*   **Latent Dimension:** 3.
*   **Activation:** Swish ($\text{Swish}(x) = x \cdot \sigma(x)$) for all hidden layers.
*   The architecture should mirror the structure shown in Fig. S2, specifically the gradually shrinking/expanding number of neurons in the hidden layers.

### Loss Functions
Training leverages two simultaneously optimized loss functions via a weighted sum (the weighting ratio will be determined empirically by the Developer to achieve the target convergence).

1.  **Reconstruction Loss ($\text{Loss}_2$):** Measures the fidelity of the reconstructed structure. Defined as the RMSD (or close proxy like MAE on superposed data) between the input coordinates and the decoder output. **Target Convergence: $\approx 1.6 \text{Å}$**.
2.  **Latent Physics Loss ($\text{Loss}_1$):** Measures the match between the 3D latent space coordinates and the input 3D Collective Variable (CV) coordinates. Defined as the RMSD/MAE on the 3D vectors. **Target Convergence: $\approx 1.0 \text{Å}$**.

## Dependencies

The required libraries for this project primarily originate from the Python scientific computing and deep learning ecosystem.

| Component/Module | Required Library | Purpose |
|:---|:---|:---|
| Deep Learning | **torch** (PyTorch) | Core deep learning framework. |
| Data Handling | **numpy** | General array and coordinate manipulation. |
| MD/Structure | **MDAnalysis** | Essential for loading/writing molecular structures, performing superposition, and calculating RMSD (crucial for accurate $\text{Loss}_2$ implementation). |
| Data Storage | **h5py** | Used for efficiently storing large coordinate/CV mock data. |
| Path Interpolation | **scipy** | Needed for B-spline or other interpolation methods to simulate the CV transition paths in the generation phase (T3.2). |
| Testing | **pytest** | Required for all unit, integration, and scientific validation tests. |
| Utilities | **tqdm, torchvision** | Progress tracking and standard deep learning dependency. |

## Tests

The QA Engineer is responsible only for tests within the `tests/` directory:

1.  **Unit Tests (Sprint 1):** Validate module-level correctness (e.g., shape of data loader output, mathematical verification of $\text{Loss}_1$ and $\text{Loss}_2$).
2.  **Integration Tests (Sprint 2):** Validate that the full system (Data $\to$ Model $\to$ Training Loop) is stable.
3.  **Scientific Validation Tests (Sprint 3):** Crucial tests to verify that the final outcome (converged losses, generated path) aligns with the scientific claims of the source paper. (AC 3.3, AC 3.4)