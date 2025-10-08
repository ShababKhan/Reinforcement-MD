# Reinforced Molecular Dynamics (rMD) Project

## Introduction
This project aims to recreate the "Reinforced Molecular Dynamics (rMD)" machine learning model as described in the scientific paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" by Kolossváry and Coffey (doi: https://doi.org/10.1101/2025.02.12.638002).

The core idea behind rMD is to overcome the limitations of traditional protein structure prediction models, which often provide static structures without accounting for dynamic molecular motions essential for biological processes. rMD addresses this by infusing physical context into a generative autoencoder. It achieves this by replacing the generic latent space of a traditional autoencoder with a physical free energy (FE) map, derived from molecular dynamics (MD) simulations.

The goal of this implemented software is to provide a functional Python-based recreation of the rMD informed autoencoder, capable of:
1.  Learning a mapping between high-dimensional protein structures and a low-dimensional physical collective variable (CV) space.
2.  Generating new, physically meaningful protein conformations by sampling points directly from the free energy map (via the CV space).

The paper demonstrates rMD's application in understanding the conformational transition of the E3 ligase substrate receptor cereblon (CRBN) from an inactive open state to an active closed state upon IMiD binding, which is a key area in targeted protein degradation.

## Methodology

This section outlines the methodology for implementing the rMD model. The project will focus on the machine learning component, consuming pre-generated data that simulates the output of MD simulations.

### rMD Model Overview
The rMD model leverages an "informed autoencoder" network. This network is trained using both protein structural data (flattened Cartesian coordinates of heavy atoms) and corresponding collective variable (CV) data derived from MD simulations. The key is a dual-loss function that ensures the autoencoder's latent space directly correlates with the physical CV space. Once trained, this allows for the generation of new protein conformations by specifying points in the CV/FE space.

### Data Preparation
For this project, we will simulate the input data that would typically come from advanced MD simulations. This includes:
*   **Protein Trajectory Data:** Flattened Cartesian coordinates of all heavy atoms for 10,000 CRBN trajectory frames. Each frame represents a protein structure superposed to a reference frame. For CRBN, the input vector length is 9696.
*   **Collective Variable (CV) Data:** Corresponding 3-dimensional CV coordinates for each trajectory frame. These CVs are defined based on the distances between the centers of mass of key CRBN domains (CRBN-NTD, CRBN-CTD, CRBN-HBD, and DDB1-BPC), particularly focusing on the conical motion of the CRBN-CTD.

### Informed Autoencoder Architecture
The autoencoder consists of an encoder and a decoder, built using fully connected (dense) layers.
*   **Encoder:** Compresses the high-dimensional protein coordinates into a low-dimensional latent space.
    *   **Layer Dimensions:** Input (9696) -> Hidden (5000) -> Hidden (1000) -> Hidden (500) -> Latent Space (3)
*   **Decoder:** Reconstructs the high-dimensional protein coordinates from the latent space.
    *   **Layer Dimensions:** Latent Space (3) -> Hidden (500) -> Hidden (1000) -> Hidden (5000) -> Output (9696)
*   **Activation Function:** The "Swish" activation function (`x * sigmoid(x)`) is applied after each hidden layer in both the encoder and decoder.

### Loss Functions & Training
The rMD autoencoder is trained using a dual-loss function approach:
*   **Loss2 (predLoss):** This loss function measures the accuracy of the structural reconstruction. It is based on the Root Mean Square Deviation (RMSD) between the input protein structure and its reconstructed output from the decoder. For superposed structures, mean absolute error or mean squared error can closely approximate RMSD for loss calculation.
*   **Loss1 (latentLoss):** This loss function ensures that the latent space of the autoencoder directly maps to the physical collective variable space. It measures the RMSD between the 3-dimensional latent space coordinates and the corresponding 3-dimensional target CV coordinates.

Both `Loss1` and `Loss2` are optimized simultaneously using a weighted sum (e.g., `total_loss = weight1 * Loss1 + weight2 * Loss2`). The training process will utilize the Adam optimizer with a batch size of 64 over 10,000 training rounds (epochs). The target values for the trained network are approximately Loss1 ≈ 1 Å and Loss2 ≈ 1.6 Å (RMSD of all heavy atoms).

### Structure Generation
Once the informed autoencoder is trained, the decoder component can be used to generate new protein conformations. By providing specific 3D CV coordinates (e.g., points sampled from a free energy map or along a desired conformational transition path), the decoder will output the corresponding high-dimensional Cartesian coordinates of the predicted protein structure.

### Post-processing (Conceptual)
The paper notes that ML-generated structures can sometimes have local geometric distortions. In the original work, these were addressed by post-processing steps using Rosetta Relax followed by position-restrained minimization. While direct integration of such external software is beyond the immediate scope of this project's ML model recreation, the conceptual need for geometry refinement is acknowledged. Potential Python-based approaches for local energy minimization or geometry correction could involve molecular mechanics libraries (e.g., OpenMM) if detailed force field information and infrastructure were to be integrated in a future extension.

## Dependencies

### Python Version
*   Python 3.9 or higher

### Libraries
The following Python libraries are required:
*   `torch` (PyTorch) - For building and training the neural network.
*   `numpy` - For numerical operations, especially for handling protein coordinates and mock data.
*   `scipy` - May be used for specific scientific utilities, though `numpy` often suffices for core operations.
*   `pytest` - For running unit tests.

### Installation
To set up your development environment, it is recommended to use a virtual environment.
1.  Create a virtual environment: `python -m venv venv`
2.  Activate the virtual environment:
    *   On Windows: `.\venv\Scripts\activate`
    *   On macOS/Linux: `source venv/bin/activate`
3.  Install the required dependencies: `pip install -r requirements.txt` (A `requirements.txt` file will be provided in the repository).

## Tests

### Testing Philosophy
This project will adopt a unit testing approach using `pytest` to ensure the correctness and reliability of individual components. Tests will cover data generation, loss functions, and the core autoencoder model functionality.

### Test Coverage
*   **Data Generation:** Verification of mock data output formats and dimensions.
*   **Loss Functions:** Validation of `rmsd_loss` and `cv_loss` calculations with known inputs.
*   **Model Components:** Basic checks for the forward pass of the encoder and decoder.

### How to Run Tests
1.  Ensure your virtual environment is activated and all dependencies are installed.
2.  Navigate to the project's root directory in your terminal.
3.  Run `pytest` to execute all tests.

### Expected Results
All tests should pass, indicating that the individual components of the rMD model are functioning as expected according to their specifications. Test outputs will provide confidence in the foundational elements before integrating them into the full training pipeline.
