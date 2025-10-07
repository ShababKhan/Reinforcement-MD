# Reinforced Molecular Dynamics (rMD) Software

## Introduction

This project aims to recreate the Reinforced Molecular Dynamics (rMD) software as described in the paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" by István Kolossváry and Rory Coffey. rMD is a machine learning-based approach that combines Molecular Dynamics (MD) simulation data with a physics-infused autoencoder to efficiently explore protein conformational space and model biologically relevant motions.

## Methodology

The rMD methodology involves the following key steps:

1.  **Free Energy Map Generation:** Conducting MD simulations biased by parameters like metadynamics or eABF along chosen Collective Variables (CVs) to generate a 3D free energy (FE) map.
2.  **Autoencoder Training:** Training a dual-loss function autoencoder. The first loss (`predLoss`, Loss2) ensures accurate reconstruction of protein structures. The second loss (`latentLoss`, Loss1) aligns the autoencoder's latent space (LS) with the CV space, effectively embedding physical information into the latent representation.
3.  **Physics-Infused Exploration:** Utilizing the trained autoencoder and the FE map to generate novel protein conformations located in low-free-energy regions of the conformational landscape. This allows for efficient exploration of biologically relevant states and transition pathways.
4.  **Structure Generation & Refinement:** Generating structures along defined pathways in the CV space and subsequently refining them to correct for local geometric distortions.


## Dependencies

The project requires the following Python libraries:

*   **Core ML Framework:** TensorFlow or PyTorch
*   **Numerical Operations:** NumPy
*   **MD Trajectory Handling:** MDAnalysis (or similar)
*   **Scientific Computing:** SciPy
*   **Visualization:** Matplotlib, PyMOL (for output rendering)
*   **Structure Refinement:** Rosetta (optional, or alternative implementation)

Please refer to `requirements.txt` for a comprehensive list of specific versions.

## Tests

Unit tests are implemented to verify the functionality of core components, including:

*   **Network Layers:** Verification of encoder and decoder architectures.
*   **Loss Functions:** Validation of `predLoss` (Loss2) and `latentLoss` (Loss1) calculations.
*   **Data Processing:** Ensuring correct data loading, preprocessing, and CV extraction.
*   **Model Training:** Testing the dual-loss training loop.
*   **Structure Generation:** Validating the generation of structures from CV inputs.
*   **Post-processing:** Confirming the application of structure refinement.

Run tests using: `pytest` (or your chosen testing framework).

## Usage

[Detailed usage instructions, including data preparation, training commands, and structure generation examples, will be provided here.]

## Contributing

[Information on how others can contribute to the project.]

## License

[License information.]
