# Reinforced Molecular Dynamics (rMD) Project Documentation

## Introduction
This document outlines the scientific methodology and structure for recreating the Reinforced Molecular Dynamics (rMD) simulation framework described in Kolossváry & Coffey (doi:10.1101/2025.02.12.638002). Our goal is to implement a dual-loss autoencoder that enforces an isomorphism between the neural network's latent space and the Free Energy (FE) landscape derived from MD collective variables (CVs).

## Methodology Summary
The rMD model uses a standard Autoencoder (AE) structure (Encoder -> Latent Space (LS) -> Decoder) trained on MD trajectory snapshots of CRBN. The key innovation is training the AE not just on structural reconstruction ($\text{Loss}_2$), but simultaneously forcing the 3D Latent Space to map isomorphically onto the 3D Free Energy (FE) landscape defined in the Collective Variable (CV) space ($\text{Loss}_1$). By minimizing $\text{Loss}_1$, the LS becomes physically meaningful, allowing us to interpret points in the LS as points on the FE map. We will generate the CRBN transition path by sampling points along a B-spline fitted across the known open/closed states in the CV space, feeding these CVs into the LS, and decoding the resulting structures.

## Dependencies
The required technology stack is **Python**. Key libraries include:
- **PyTorch/TensorFlow:** For building and training the neural network. (We default to PyTorch for modern implementations unless specified otherwise).
- **NumPy:** For numerical array operations.
- **SciPy:** For mathematical functions, particularly B-spline fitting.
- **MDAnalysis/Biopython:** Required for structure parsing and calculating domain Center of Masses (COMs) for CV generation.

*Note: Post-processing relies on external tools like Rosetta, as mentioned in the paper.*

## Tests
All code must adhere to **PEP 8** standards. The primary verification points are:
1.  Final converged $\text{Loss}_2$ (Reconstruction RMSD) $\approx 1.6 \text{ Å}$.
2.  Final converged $\text{Loss}_1$ (Latent Space Alignment) $\approx 1.0 \text{ Å}$.
3.  The 3D Latent Space must be topologically consistent with the 3D Free Energy map.