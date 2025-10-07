# Reinforced Molecular Dynamics (rMD) Python Package

This package provides a Python implementation of the Reinforced Molecular Dynamics (rMD) technology, as described in "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" by Kolossváry and Coffey (doi: https://doi.org/10.1101/2025.02.12.638002).

rMD combines molecular dynamics (MD) trajectory data and free-energy (FE) map data to train a dual-loss function autoencoder network, enabling efficient exploration of conformational space and targeted protein structure generation.

## Introduction
*(This section will be filled out by the team during development)*
A detailed introduction to the rMD methodology, its scientific significance, and the biological problems it aims to address, particularly in the context of protein conformational changes like the CRBN open-to-closed transition.

## Methodology
*(This section will be filled out by the team during development)*
An in-depth explanation of the algorithms and scientific methods implemented in this package:
-   **Molecular Dynamics (MD) Simulation Data Preprocessing:** Details on loading trajectory data, superposition algorithms, extraction of CRBN heavy atom coordinates (length-9696 vectors), and dataset splitting (80/20 train/validation).
-   **Collective Variable (CV) Definition and Calculation:** Precise definition and implementation of the three CVs (distances between COMs of CRBN-NTD, CRBN-CTD, CRBN-HBD, and DDB1-BPC) as per Figure S1.
-   **Free Energy (FE) Map Integration:** Description of how the pre-computed 3D free energy map is loaded and utilized.
-   **Informed Autoencoder Network Architecture:** Detailed breakdown of the Encoder and Decoder architectures, including layer dimensions (as per Figure S2), fully connected layers, and Swish activation functions.
-   **Dual-Loss Function Formulation:** Mathematical and conceptual explanation of Loss1 (latentLoss, relating latent space to CVs using RMSD/MAE) and Loss2 (predLoss, reconstruction error using MAE), and their weighted combination.
-   **Training Procedure and Hyperparameters:** Description of the training loop, Adam optimizer, batch size (64), and training rounds (10,000 epochs).
-   **Protein Structure Generation and Path Exploration:** Explanation of how new structures are generated from CVs, B-spline fitting for defining conformational paths, and the generation of atomistic trajectories along these paths.
-   **Post-processing of Generated Structures:** Discussion of the necessity for external structural refinement (e.g., Rosetta Relax, minimization) to address geometric distortions, and how this process is integrated or recommended.

## Dependencies
*(This section will be filled out by the team during development)*
A comprehensive list of all required Python libraries (e.g., `numpy`, `scipy`, `torch`/`tensorflow`, `MDAnalysis`, `tqdm`, `matplotlib`) and any external software (e.g., for post-processing) with their recommended versions.

## Installation
*(This section will be filled out by the team during development)*
Clear and concise instructions on how to install and set up the rMD Python package, including any environment configuration steps.

## Usage
*(This section will be filled out by the team during development)*
Detailed examples and tutorials demonstrating how to use the rMD package for:
-   Preparing MD trajectory data.
-   Training the informed autoencoder with custom datasets.
-   Generating new protein structures from specific CV coordinates.
-   Defining and exploring conformational transition paths.
-   Visualizing results (latent space, FE maps, generated structures).

## Tests
*(This section will be filled out by the team during development)*
Details on the testing procedures employed to ensure code quality and scientific accuracy, including:
-   Unit tests for individual functions and modules.
-   Integration tests for end-to-end workflows (e.g., data prep to structure generation).
-   Scientific validation tests to compare generated results (e.g., Loss1/Loss2 values, structural quality) against benchmarks from the paper.

## Code Standards
All code developed for this project must strictly adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.

## Documentation Standards
All documentation, including inline comments and docstrings, must be clear, comprehensive, and up-to-date, following standard Python documentation practices.
