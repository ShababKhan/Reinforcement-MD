# rMD: Reinforced Molecular Dynamics Python Package

## Introduction
This package implements the Reinforced Molecular Dynamics (rMD) method, a physics-infused generative machine learning model for exploring biologically relevant protein motions, as described in "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" by Kolossváry and Coffey (doi: https://doi.org/10.1101/2025.02.12.638002).

rMD combines molecular dynamics (MD) trajectory data and free-energy (FE) map data to train a dual-loss function autoencoder network. The key innovation is replacing the latent space of a standard autoencoder with a physically meaningful free-energy map, computed over a low-dimensional collective variable (CV) space. This allows the network to explore conformational space more efficiently and generate targeted protein structures in biologically relevant regions or along conformational transition paths.

## Methodology
The core methodology of rMD involves:
1.  **Data Preparation**: Molecular Dynamics (MD) trajectory frames (Cartesian coordinates of selected atoms, e.g., heavy atoms of CRBN) are collected and preprocessed (superposition, flattening). Simultaneously, a set of Collective Variables (CVs) describing the conformational changes is defined, and their values are computed for each trajectory frame. A 3-dimensional Free Energy (FE) map, derived from advanced sampling MD simulations along these CVs, is also required as input.

2.  **Informed Autoencoder Network**: A neural network, consisting of an encoder and a decoder, is constructed using PyTorch. The encoder maps high-dimensional protein structures to a low-dimensional latent space (3D in this work), and the decoder reconstructs structures from this latent space. The network architecture consists of fully connected layers with Swish activation functions (except for the latent space and output layers).

3.  **Dual-Loss Training**: The autoencoder is trained using a novel dual-loss function approach:
    *   **Loss1 (Latent Space Loss)**: Measures the Root Mean Square Distance (RMSD) between the autoencoder's latent space coordinates and the corresponding Collective Variable (CV) coordinates. This loss physically anchors the latent space.
    *   **Loss2 (Prediction Loss)**: Measures the Root Mean Square Distance (RMSD) between the input protein structures and their reconstructed counterparts from the decoder. This ensures the structural integrity of the generated outputs.
    Both losses are optimized concurrently using an optimizer like Adam.

4.  **Structure Generation**: Post-training, the autoencoder can generate new protein structures. By providing specific points or paths directly from the pre-computed Free Energy (FE) map (which is now correlated with the latent space), the decoder can predict atomic-level protein structures that correspond to desired free energy regions or conformational transition pathways.

5.  **Post-processing (External)**: Generated structures may contain local geometric distortions. It is recommended to apply external structural relaxation techniques such as Rosetta Relax followed by position-restrained minimization to refine the predicted conformations. This step is outside the scope of direct implementation within this Python package but is crucial for obtaining high-quality structures.

## Dependencies
This project requires Python 3.9+ and the following libraries:
*   `numpy` (>=1.20.0): For numerical operations.
*   `scipy` (>=1.7.0): For scientific computing, including B-spline interpolation.
*   `torch` (>=1.10.0): The PyTorch deep learning framework.
*   `MDAnalysis` (>=2.0.0): For loading, processing, and manipulating molecular dynamics trajectory data.
*   `tqdm` (>=4.0.0): For displaying progress bars (optional).
*   `scikit-learn` (>=1.0.0): Potentially for robust RMSD calculations.

## Tests
This section will document the testing procedures and results, ensuring the scientific accuracy and code quality of the rMD implementation.
*   **Unit Tests**: Comprehensive tests for individual functions and classes (e.g., CV calculation, loss functions, encoder/decoder forward passes).
*   **Integration Tests**: Tests to verify the complete data preprocessing, training, and structure generation pipelines.
*   **Validation**:
    *   Comparison of achieved `Loss1` and `Loss2` values against those reported in the paper (e.g., Loss1 ≈ 1 Å, Loss2 ≈ 1.6 Å).
    *   Visual inspection and RMSD comparison of generated structures against known conformations or MD frames.
    *   Verification of the one-to-one correspondence between latent space and CV space (e.g., by plotting).

---
**Code Quality Standard**: All code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards.
**Documentation Standard**: All public functions, classes, and modules must have clear, concise, and comprehensive docstrings.
