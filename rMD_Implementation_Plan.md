# Reinforced Molecular Dynamics (rMD) Implementation Plan

This document serves as the foundational Project Blueprint for recreating the Reinforced Molecular Dynamics framework described in the paper: "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" (Kolossváry & Coffey, bioRxiv 2025).

All agents must adhere to this plan. The final code must conform to **PEP 8** standards.

---

## 1. Methodology Summary

The objective is to replicate the **Informed Autoencoder** training pipeline using Python. The core innovation is the dual-loss training that links the network's Latent Space (LS) to the Free Energy (FE) landscape defined by Collective Variables (CVs).

**Key Steps:**
1.  Obtain/Mock 10,000 CRBN structure/CV pairs.
2.  Build a 3D Latent Space Autoencoder using fully connected layers and **Swish** activation.
3.  Train simultaneously on **Loss2 (Reconstruction RMSD)** and **Loss1 (LS to CV RMSD)**.
4.  Use the trained model to map a B-spline path defined in CV space back into conformational structures.

## 2. Component & Dependency List

| Component | Technology/Library | Rationale |
| :--- | :--- | :--- |
| **Core ML Framework** | **TensorFlow/Keras** or **PyTorch** | To build and train the deep neural network (Autoencoder). |
| **Numerical Ops/Data Handling** | `numpy` | Essential for array manipulation of Cartesian coordinates and CVs. |
| **Structure/Fitting Utility** | `scipy.interpolate` | Needed to implement the B-spline fitting logic for path generation. |
| **Data Generation Utility** | (To be written) | Script to load mock data matching required structure vector length (9696 for all heavy atoms) and 3 CVs. |
| **Post-Processing** | *External Reliance* | **Rosetta Relax** is assumed for final structure refinement (S14). |

## 3. Agile Project Plan (Scrum Tasks)

### **Sprint 1: Data & Architecture Setup**

| Task ID | User Story | Details | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **T1.1** | Data Setup | Create mock data generation script (`generate_mock_data.py`). Must implement structure superposition logic for input preparation per S03. | Script generates two arrays: Structures (10000, 9696) and CVs (10000, 3). |
| **T1.2** | Model Shell | Implement Autoencoder base model structure. | Model is symmetric, has 3D latent space, uses Swish activation on hidden layers. |
| **T1.3** | Loss Implementation | Implement Loss2 function (RMSD). | `Loss2_RMSD(struct_in, struct_out)` function defined. |

### **Sprint 2: Physics Infusion & Training**

| Task ID | User Story | Details | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **T2.1** | Physics Loss | Implement Loss1 function (LS to CV mapping). | `Loss1_CV_Mapping(LS_coords, CV_targets)` function defined. |
| **T2.2** | Dual-Loss & Training | Implement the combined loss and training loop configuration. | Combined loss function handles weighted sum. Training loop configuration matches S11 (Adams, Batch 64, 10k rounds). |
| **T2.3** | Training Execution | Execute initial training runs. | Final Loss1 $\approx 1.0 \text{ Å}$ and Loss2 $\approx 1.6 \text{ Å}$ are recorded/logged (S12). |

### **Sprint 3: Path Exploration & Validation**

| Task ID | User Story | Details | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **T3.1** | Path Logic | Implement CV path generation logic (B-spline fit approximation). | Function generates a sequence of smooth CV points along a defined path. |
| **T3.2** | Structure Generation | Implement structure generation from CV path using the decoder. | Structures are successfully decoded from the generated CV path points. |
| **T3.3** | Documentation Finalization | Create this master documentation file (`rMD_Implementation_Plan.md`). | This file (and its structure) is finalized and committed. |
| **T3.4** | Final Acceptance Check | Comprehensive review against the Project Verification Checklist (S01-S14). | All scientific details are mapped to implemented tasks. |

## 4. Methodology Verification Checklist (Internal Reference)

*(QA Engineer to track fulfillment)*

| ID | Paper Element | Status | Mapped Task(s) |
| :--- | :--- | :--- | :--- |
| S01 | CRBN Transition focus | Unaddressed | N/A (Goal Defined by Project) |
| S02 | MD Simulation: meta-eABF method | Addressed (As Precursor) | T1.1 (Mocking Input) |
| S03 | Input Data: System prep, superposition | Addressed | T1.1 |
| S04 | System Prep Details (Force fields, Temp) | Addressed (As Config Context) | T1.1 |
| S05 | CV Definition: 3D space based on COMs | Addressed (As Input Data Structure) | T1.1 |
| S06 | Autoencoder Structure: FC layers, Swish | Addressed | T1.2 |
| S07 | Latent Space Dimension: 3D | Addressed | T1.2 |
| S08 | Training Data Size: 10k frames (8k/2k) | Addressed | T1.1 |
| S09 | Loss2 Calculation (RMSD) | Addressed | T1.3 |
| S10 | Loss1 Calculation (LS to CV) | Addressed | T2.1 |
| S11 | Training Optimizer/Config (Adams, 10k rounds) | Addressed | T2.2 |
| S12 | Target Loss Values (L1 $\approx 1$, L2 $\approx 1.6$) | Addressed | T2.3 |
| S13 | Path Generation: B-spline on CV space | Addressed | T3.1 |
| S14 | Structure Post-Processing: Rosetta Relax | Noted (External) | T3.2 (Requires subsequent steps) |
