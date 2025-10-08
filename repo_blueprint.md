# Project Blueprint: Reinforced Molecular Dynamics (rMD) for CRBN Conformational Analysis

**Project Goal:** To faithfully replicate the Reinforced Molecular Dynamics (rMD) methodology described in Kolossváry & Coffey (2025) by training a dual-loss autoencoder informed by a simulated Free Energy map of the CRBN open-close transition.

## Introduction
This project aims to recreate the rMD simulation pipeline. The primary deliverable is a Python-based implementation of the informed autoencoder capable of generating high-quality, physically constrained protein conformations, specifically demonstrating the CRBN open-to-closed structural transition. All code must adhere strictly to **PEP 8** standards.

## Methodology
This section details the step-by-step computational pipeline, cross-referencing the tasks defined in the Agile Project Plan.

### 1. Input Data Preparation (Simulated)
*   **System:** CRBN E3 Ligase (using structural information from PDB: 6H0F and 6H0G).
*   **Simulation Method:** Meta-eABF simulations are required to generate trajectory data and the 3D Free Energy Map.
    *   *Dependencies:* OpenMM, Plumed.
    *   *Conditions:* 310 K, NPT, HMR ($\text{H mass}=3.024 \text{ amu}$), 4 fs timestep.
*   **Collective Variables (CVs):** Three distances defined by the Center of Mass (COM) of CRBN-NTD, CRBN-CTD, and CRBN-HBD relative to DDB1-BPC (Figure S1).
*   **Training Data:** 10,000 heavy-atom coordinate sets for CRBN, pre-processed by superposition to the first frame.

### 2. Informed Autoencoder Network
*   **Architecture:** Encoder/Decoder chain with fully connected layers and **Swish** activation ($x\sigma(x)$).
*   **Latent Space:** 3-Dimensional ($D_{LS}=3$), corresponding directly to the 3 CVs.
*   **Loss Functions:**
    *   **Loss 2 (Reconstruction):** Average heavy-atom RMSD ($\text{Target} \approx 1.6 \text{ Å}$).
    *   **Loss 1 (Physics Infusion):** Alignment loss between Latent Space coordinates and true CV coordinates.
*   **Training:** Adam optimizer, 10,000 rounds, Batch size 64.

### 3. Transition Path Generation & Post-Processing
*   **Path Definition:** Use B-spline fitting on manually defined anchor points across the established 3D FE map (Figure 4).
*   **Prediction:** Feed B-spline points (in CV space) through the trained decoder.
*   **Relaxation:** Post-process predicted structures using logic mimicking **Rosetta Relax** followed by C$\alpha$/C$\beta$ restrained minimization to correct local distortions.
*   **Visualization:** Generate transition movie and Pymol session files.

## Dependencies
| Category | Libraries/Tools | Python Environment |
| :--- | :--- | :--- |
| **Core ML** | TensorFlow or PyTorch, NumPy | Essential |
| **Simulation Handling** | MDAnalysis (or similar coordinate parser) | Recommended |
| **Wolfram Logic Replication** | (N/A - Logic must be translated to Python/TF/PyTorch) | Critical |

*Note: The simulation data generation (OpenMM/Plumed) will be mocked or stubbed if immediate execution is not feasible, focusing on the data format required by the ML model.*

## Tests
*   **T0.1:** Successful generation of 10,000 superposed heavy-atom coordinate sets (80/20 split).
*   **T1.3:** Basic Autoencoder verifies reconstruction quality ($\text{Loss}_2 \approx 1.2 \text{ Å}$).
*   **T2.3:** Final convergence shows $\text{Loss}_2 \approx 1.6 \text{ Å}$ alongside acceptable $\text{Loss}_1$.
*   **T2.2:** Generated path coordinates in the LS space visually align with the shape of the input FE map (as shown in Figure 3).
*   **T2.3:** Post-processed structures successfully generated, representing the CRBN transition states.