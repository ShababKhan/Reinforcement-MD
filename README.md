# Reinforced Molecular Dynamics (rMD) Replication Project

This project aims to replicate the Reinforced Molecular Dynamics (rMD) methodology as described in the paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process."

The core objective is to create a robust, accurate, and well-documented software implementation of the informed autoencoder capable of generating physically relevant protein conformations.

## Introduction

This project aims to digitally replicate the scientific software and methodology detailed in the paper, *Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process*. Our focus is on recreating the **rMD framework**, which uses a dual-loss autoencoder to establish a physical link between its latent space and the collective variable (CV) space of the molecular system. This allows for targeted generation of protein structures, specifically modeling the open-to-closed conformational transition of the CRBN protein.

## Methodology

The rMD approach is centered on making the autoencoder's latent space physically interpretable.

The key steps in the implementation process are:

1.  **Data Preparation (U1):** Load MD trajectory data for CRBN (simulated/mock), extract only the heavy-atom Cartesian coordinates of CRBN, flatten them into a $\sim 9696$-length feature vector, and ensure all structures are superposed to a reference frame. This requires 8000 training and 2000 validation structures.
2.  **Collective Variable (CV) Calculation (T1):** Implement functions to derive the 3-dimensional CV vector ($CV_1, CV_2, \text{ and } CV_3$). These CVs are defined by the distances between the Center of Mass (COM) of the CRBN-CTD domain and the COMs of the three reference regions: CRBN-NTD, CRBN-HBD, and DDB1-BPC (as per Figure S1).
3.  **Model Definition (T2):** Design the feed-forward autoencoder architecture using fully connected layers and the Swish activation function, with a 3-dimensional latent space.
4.  **Dual-Loss Training (U2):** Train the autoencoder (10,000 rounds, batch size 64, Adam optimizer) using a weighted sum of two loss functions:
    *   **Loss2 (Prediction Loss):** Minimizes the reconstruction RMSD between the input structure and the decoded output (target $\approx 1.6 \text{\AA}$).
    *   **Loss1 (Latent Loss):** Minimizes the RMSD between the 3D latent space coordinates and the target 3D CV coordinates, thereby forcing the LS to adopt the physical CV space (target $\approx 1.0 \text{\AA}$).
5.  **Structure Generation (U3, T4):** Utilize the trained model's decoder to generate new structures. This involves using B-spline interpolation in CV space (which now corresponds to the LS) to define a transition path and feeding these interpolated CV points to the decoder to predict the full atomistic conformations.

## Dependencies

The project must be implemented in Python, adhering to PEP 8 standards. The required third-party libraries include:

| Dependency | Purpose | Equivalent Python Library |
| :--- | :--- | :--- |
| **Deep Learning Framework** | Core autoencoder architecture and training. | `PyTorch` |
| **Numerical Operations** | Handling multi-dimensional arrays, matrix operations, loss function calculation. | `NumPy` |
| **Structure Handling/Analysis** | Loading PDB/Trajectory data, calculating COM, and performing superposition/alignment. | `MDAnalysis` (Highly Recommended) |
| **B-Spline Interpolation** | Path generation in CV space. | `SciPy` (`scipy.interpolate`) |
| **Data Handling** | General data manipulation and mock data generation. | `Pandas` (Optional for data processing) |
| **Progress Reporting** | Monitoring training progress. | `tqdm` |

## Tests

This section will be populated by the QA Engineer with details of the testing framework and validation results for each sprint.

### Sprint 1 Tests (Q1 - Data Pipeline Test Suite)

*To be populated by the QA Engineer.* The initial focus will be on validating the data preparation pipeline (`U1`) and the CV calculation module (`T1`). Validation benchmarks include:

*   **CV Calculation Logic:** Verify that the logic accurately defines the Collective Variables based on the COM distances of the four specified domains/regions.
*   **Data Structure:** Confirm that the input vector length is 9696 (heavy atoms of CRBN).
*   **Data Split:** Confirm the 8000 training / 2000 validation split.

### Sprint 2 Tests (Q2 - Model Validation and Gaps Check)

*To be populated by the QA Engineer.* The focus will be on validating the core rMD implementation:

*   **Model Performance (Losses):** Verification that Loss1 $\approx 1.0 \text{\AA}$ and Loss2 $\approx 1.6 \text{\AA}$ are achieved.
*   **Physics Infusion:** Demonstrate a strong, quantifiable correlation between the 3D Latent Space coordinates and the 3D CV original data for the validation set, confirming that the physical context is successfully infused.
*   **Path Generation:** Verify the B-spline interpolation accurately defines a continuous path in CV space.
