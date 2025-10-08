# rMD Test Plan

## 1. Overview
This test plan covers the scientific validation and quality assurance for the Reinforced Molecular Dynamics (rMD) software package, designed to model the CRBN open-to-closed conformational transition. The primary goal is to ensure the implemented code faithfully reproduces the scientific claims and numerical results reported in the preprint "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" (Kolossváry & Coffey, 2025).

## 2. Test Strategy
The testing strategy is composed of:
*   **Unit Tests:** Verifying the mathematical correctness of utility functions (e.g., CV calculation, coordinate manipulation).
*   **Integration Tests:** Verifying the sequential pipeline components (e.g., simulation output correctly formatted for autoencoder training).
*   **Scientific Validation Tests (Critical):** Executing the full rMD pipeline (or key parts thereof) against known external data and verifying the final numerical/structural outputs against paper benchmarks.
*   **Performance Benchmarks:** Sanity-checking training time against the reported 2 hours on an RTX 4080 GPU (qualitative check only, as hardware will differ).

## 3. Test Data Fixtures

### Required PDB Structures (for manual checks/starting points)
*   `6H0G`: Closed CRBN X-ray structure (Anchor point validation).
*   `6H0F`: Open CRBN X-ray structure (Anchor point validation).

### Mock Data Requirements
*   **Trajectory Frames:** Mock NumPy arrays representing flattened Cartesian coordinates (length 9696) for 10,000 frames (8000 training / 2000 validation split). These must be structured to test the reshaping/superposition logic.
*   **CV Coordinates Mock:** Mock array of shape (10000, 3) representing the target 3D CV space for `Loss1` targets.
*   **Free Energy Map Mock:** Mock data structure representing the target 3D FE map for sanity checks in structure generation interpolation.

## 4. Validation Thresholds (Derived from Paper)
| Metric | Paper Value (Target Context) | Tolerance (Acceptance Criteria) | Validation Module |
| :--- | :--- | :--- | :--- |
| **Loss1** (CV RMSD) | approx 1.0 A | Target +/- 10% | `test_training_loss1` |
| **Loss2** (Heavy Atom RMSD) | approx 1.6 A | Target +/- 10% | `test_training_loss2` |
| **Latent Space Dimensionality** | 3 | Exact Match | `test_autoencoder_architecture` |
| **Input Vector Size** | 9696 | Exact Match | `test_data_preparation` |
| **Training Time** | 2 hours (RTX 4080) | Qualitative Pass (Must complete in reasonable time) | `test_performance` |
| **CV Map Feature** | Narrow path (low free energy) between Open (6H0F) and Closed (6H0G) states. | Visual confirmation / CV distance check between anchor points. | `test_validation_cv_map` |

## 5. Test Coverage Requirements
All acceptance criteria documented in the `PROJECT_BLUEPRINT.md` will be mapped to individual unit or integration tests. Emphasis will be placed on tests verifying:
*   Correct superposition of input structures (`data_preparation.py`).
*   Correct mapping of LS coordinates to CV coordinates (`training.py` / `test_cv.py`).
*   Accurate reconstruction of input structures (`autoencoder.py`).
*   Generation of structures along the B-spline path defined in Figure 4.

## 6. Test Execution Plan
*   **Unit Tests:** Executed on every pull request via CI (once set up).
*   **Integration/Validation Tests:** Executed manually during Sprint review cycles. Full pipeline testing depends on successful completion of Sprints 1-6.
