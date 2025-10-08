# rMD Project Master Documentation

This document charts the implementation progress, linking the scientific methodologies described in the Kolossváry & Coffey bioRxiv preprint to the developed Python modules.

## Methodology

The core of the Reinforced Molecular Dynamics (rMD) approach centers on training an **Informed Autoencoder** that uses a physics-based constraint ($\mathcal{L}_1$) in addition to the standard structural reconstruction constraint ($\mathcal{L}_2$).

| Component Implemented | Paper Reference | Implementation Detail | Source File |
| :--- | :--- | :--- | :--- |
| **Data Preparation** | *"...all protein structures were superposed to the first frame..."* (Para. 10, Informed AE Network) | Implemented `superpose_structures` to enforce a reference frame alignment for the 9696-dimensional heavy atom input vectors. | `src/data_utils.py` |
| **Informed Autoencoder Architecture** | Figure 2, Figure S2, *"...element-wise activation layer using the 'Swish' form..."* | Implemented a symmetric Encoder/Decoder structure (`InformedAutoencoder`) with an intermediate 3D Latent Space (LS) and mandated Swish activation functions across hidden layers. | `src/model_architecture.py` |
| **$\mathcal{L}_2$ (Prediction Loss)** | Loss2 function; minimizing average RMSD (Para. 8, Figure 2) | Implemented as Mean Squared Error (MSE) over the input/output coordinates in `calculate_loss2_rmsd`. | `src/losses.py` |
| **$\mathcal{L}_1$ (Latent Loss)** | Loss1 function; LS coordinates $\rightarrow$ CV coordinates (Para. 10, Figure 2) | Implemented as MSE between latent space output and target CV coordinates in `calculate_loss1_latent_cv_rmsd`. This injects the physical context. | `src/losses.py` |
| **Dual-Loss Training** | *"...simultaneously optimizing Loss1 and Loss2, e.g., using a weighted sum..."* | Implemented `DualLoss` class to calculate $\alpha \mathcal{L}_1 + \mathcal{L}_2$. The `train_r_model` script uses the Adam optimizer for 10,000 rounds, batch size 64. | `src/train.py` |

## Sprint 1 Deliverables Summary
- **Status:** Complete.
- **Artifacts:** `src/data_utils.py`, `src/model_architecture.py`, `src/losses.py`, `src/train.py`, `tests/test_s1_core.py`.

## Next Steps (Sprint 2)
The focus shifts to using the trained model to generate the transition path as described in **Figure 4** of the paper.

### Sprint 2 Tasks (Advanced Functionality & Path Generation)

| Task ID | User Story | Acceptance Criteria (AC) | Status |
| :--- | :--- | :--- | :--- |
| **S2.T1** | Implement B-Spline Path Generation Utility | AC 1.1: Function accepts anchor CV points and generates a set of intermediate points along the fitted B-spline path (mimicking Fig. 4 blue curve). | To Do |
| **S2.T2** | Implement Structure Generation Endpoint (CV $\rightarrow$ Structure) | AC 2.1: A public function exists to use the trained model to generate a full atomic structure from candidate CV coordinates. | To Do |
| **S2.T3** | Execute Full Transition Path Generation | AC 3.1: Run the path generated in S2.T1 through the trained model to produce the full set of transition structures. | To Do |
| **S2.P1** | Final Documentation & External Tool Noting | AC 5.1: `DOCUMENTATION.md` is finalized. AC 5.2: Clear dependency instructions are added for the external post-processing steps (Rosetta/Pymol). | To Do |

**Action Required:** Software Developer Agent to commence work on **S2.T1** and **S2.T2**. QA Engineer to prepare tests for path generation logic.