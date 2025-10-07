# Reinforced Molecular Dynamics (rMD) Implementation

## Introduction

This project aims to implement and provide a computational framework for Reinforced Molecular Dynamics (rMD) as described in the paper "[Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process](doi: https://doi.org/10.1101/2025.02.12.638002)" by István Kolossváry and Rory Coffey. The rMD technique combines Molecular Dynamics (MD) simulation data with a physics-informed autoencoder to efficiently explore protein conformational space and model dynamic processes.

## Methodology

The core of this implementation follows the rMD methodology:

1.  **MD Simulation & Free Energy (FE) Map Generation:** (Details on how to acquire or generate FE maps and MD trajectories, including Collective Variable (CV) definitions)
    *   Collective Variable (CV) definitions (e.g., based on atom/residue COM distances).
    *   Free Energy (FE) map computation using methods like meta-eABF.
2.  **Informed Autoencoder Training:**
    *   **Architecture:** Encoder-decoder network with a latent space.
    *   **Loss Functions:**
        *   `Loss2` (Reconstruction Loss): Minimizes RMSD between input and reconstructed structures.
        *   `Loss1` (Latent-to-CV Loss): Minimizes distance between latent space coordinates and CV coordinates.
    *   **Training Process:** Simultaneous optimization of `Loss1` and `Loss2` to map the latent space to the physical FE landscape.
3.  **Structure Generation & Exploration:**
    *   Generating protein structures from latent space or CV coordinates.
    *   Exploring conformational transitions by defining paths on the FE map and reconstructing corresponding structures.
4.  **Post-processing:**
    *   Relaxation of generated structures using tools like Rosetta Relax.

## Dependencies

This project requires the following software and libraries:

*   **Python:** Version 3.x
*   **Core Libraries:**
    *   NumPy
    *   Pandas (optional, for data handling)
    *   TensorFlow or PyTorch (specify which one)
    *   Scikit-learn (for potential utility functions)
*   **Visualization:**
    *   Matplotlib
    *   Seaborn
    *   (Optional) PyMOL (for advanced visualization and movie generation)
*   **Structural Biology Tools:**
    *   MDAnalysis or Biopython (for PDB/trajectory file handling and analysis)
*   **External Software (if reproducing simulations):**
    *   OpenMM
    *   PLUMED
    *   Rosetta (for structure relaxation)

## Installation

(Detailed instructions on setting up the environment, installing dependencies, and downloading/preparing data.)

*   Clone the repository.
*   Create a virtual environment (e.g., using `venv` or `conda`).
*   Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
*   Instructions for installing external software (OpenMM, PLUMED, Rosetta) if applicable for full reproduction.

## Usage

(Detailed examples on how to run the code.)

*   **Data Preparation:** How to format MD trajectory data and FE maps.
*   **Training the Informed Autoencoder:** Command-line arguments or script to trigger training.
*   **Generating Structures:** Using the trained model to predict structures from CV or latent space coordinates.
*   **Exploring Conformational Transitions:** Defining paths and generating transition movies.
*   **Running Simulations (Optional):** Instructions for setting up and running meta-eABF simulations using OpenMM/PLUMED.

## Contribution

(Guidelines for contributing to the project.)

## License

(Specify the license.)

## Acknowledgements

Based on the work by István Kolossváry and Rory Coffey.
