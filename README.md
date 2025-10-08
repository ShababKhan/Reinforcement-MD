# Reinforced Molecular Dynamics (rMD) Reimplementation

**Project Goal:** To accurately recreate the physics-infused generative machine learning model described in I. Kolossváry & R. Coffey's bioRxiv preprint, focusing on modeling the CRBN open-to-closed conformational transition.

All code must strictly adhere to **PEP 8 standards**. All documentation must be clear, precise, and reflect the current implementation status.

---

## Introduction
[To be filled by the development team after initial setup. Should summarize the paper's context on molecular glue degraders and the need for dynamic modeling.]

## Methodology
This section will detail the implementation steps corresponding to the paper's key methods:

### 1. MD Simulation and CV Definition
*   **System:** DDB1-CRBN apo complex (PDB IDs 6H0F/6H0G).
*   **Method:** Meta-eABF simulation to generate the 3D Free Energy Landscape.
*   **CV Definition:** Implementation details for the 3 Collective Variables derived from the COM distances (CRBN-NTD, CTD, HBD, DDB1-BPC).

### 2. Informed Autoencoder (rMD Network)
*   **Architecture:** Implementation of the Encoder-Decoder structure with a 3-dimensional latent space.
*   **Loss Functions:** Details on the implementation of $\text{Loss}_2$ (RMSD reconstruction) and $\text{Loss}_1$ (Latent Space to CV mapping).
*   **Training:** Configuration of the Adams optimizer, batch size, and Swish activation functions.

### 3. Transition Path Generation
*   **Path Definition:** Implementation of B-spline fitting in the CV space to define the transition path between open and closed states.
*   **Structure Generation:** Method used to map CV points from the B-spline onto the trained decoder network.
*   **Post-processing:** Details on structure relaxation using external tools (Rosetta/local minimizer).

## Dependencies
A comprehensive list of all required software and Python libraries, including specific versions where necessary for reproducibility.

*   **Core Python Libraries:** NumPy, SciPy, PyTorch/TensorFlow, MDAnalysis (to be confirmed).
*   **Simulation/External Tools:** OpenMM, Plumed, AmberTools, (External dependency for structure refinement: Rosetta).

## Tests
This section will document the validation strategy, mapping directly to the Project Blueprint's Acceptance Criteria (AC) and the Verification Checklist (C1-C25).

*   **Unit Tests:** Tests for CV calculation accuracy and Loss function calculations ($\text{Loss}_1, \text{Loss}_2$).
*   **Integration Tests:** Full pipeline run targeting $\text{Loss}_1 \approx 1.0$, $\text{Loss}_2 \approx 1.6$.
*   **Validation Tests:** Comparison of the final generated structures and path topology against Figure 4 of the paper.

---
**Status:** Initial Blueprint Complete. Awaiting Sprint 1 execution. ### Agile Plan Summary: 4 Sprints planned to cover Data Prep, ML Model, Application, and Validation.