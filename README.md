# Reinforcement Molecular Dynamics (rMD) Simulation Project

This repository aims to recreate the Reinforced Molecular Dynamics (rMD) technology described in the bioRxiv preprint: "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" by Kolossváry and Coffey.

## Project Status: Sprint 1 In Progress
**Focus:** Establishing the core Machine Learning pipeline (Data Utility, Model Architecture, Dual-Loss Functions) based on Paper Section: Informed Autoencoder Network.

### Implemented Files:
- `src/data_utils.py`: Implements mock data generation and structure superposition for the 9696-dimension CRBN input vectors.
- `src/model_architecture.py`: Defines the symmetric Informed Autoencoder structure with 3D latent space and Swish activations.
- `src/losses.py`: Implements the core `Loss1` (Latent-CV distance) and `Loss2` (Reconstruction RMSD) criteria, combined into `DualLoss`.
- `src/train.py`: Implements the dual-loss training loop using the Adam optimizer, matching paper hyper-parameters (10k rounds, Batch 64).
- `tests/test_s1_core.py`: Unit tests covering data utility functions and loss calculations.

### Plan Summary (Goals for Sprint 1: Core Model Implementation)
The core components required for training physics-infused autoencoders have been modeled and tested successfully against the specifications:
- **S1.T1 (Data Pipeline):** Complete. (Using mock data matching dimensions).
- **S1.T2 (Architecture):** Complete. (Symmetric AE with Swish and 3D LS).
- **S1.T4/T5 (Losses):** Complete. (Dual-loss defined).
- **S1.T6 (Training):** Complete. (Training loop structure implemented).
- **S1.T7 (Testing):** Complete. (Core component tests implemented).

**Next Step:** Execute the training loop (`src/train.py`) on mock data and formally assign Sprint 2 tasks to the development team.