# Reinforced Molecular Dynamics (rMD)

## Introduction

Reinforced Molecular Dynamics (rMD) is a physics-infused generative machine learning framework for modeling biologically relevant protein conformational transitions. rMD combines molecular dynamics trajectory data with free-energy maps to train a dual-loss autoencoder network that explores conformational space more efficiently than traditional MD simulations.

### Key Features
- **Physics-informed latent space**: Directly links neural network latent space to collective variable space via free energy maps
- **Self-contained**: No reliance on pre-trained models; trains from your own MD data
- **Desktop-friendly**: Runs on a single GPU workstation
- **Dual-loss architecture**: Simultaneously optimizes reconstruction accuracy and physics-based constraints
- **Targeted structure generation**: Generate protein conformations in specific regions of conformational space

### Scientific Background
rMD addresses the challenge of modeling protein dynamics by replacing the traditional autoencoder's abstract latent space with a physically meaningful free-energy landscape. This approach enables:
- Exploration of poorly sampled conformational regions
- Prediction of conformational transition pathways
- Generation of biologically relevant structural ensembles

The method was developed and validated on the cereblon (CRBN) E3 ligase substrate receptor, successfully modeling the open-to-closed conformational transition induced by immunomodulatory imide drugs (IMiDs).

## Methodology

### 1. System Preparation
PDB structures are prepared for simulation through a multi-step process:
- Loading of protein structures (e.g., CRBN-DDB1 complex)
- Addition of missing residues using computational methods
- Application of force fields (ff14sb for proteins, gaff for small molecules, TIP3P for water)
- Hydrogen mass repartitioning (HMR) to enable longer timesteps
- Setup of periodic boundary conditions with PME electrostatics

### 2. Collective Variable Definition
Biologically relevant collective variables (CVs) are defined based on center-of-mass (COM) distances between protein domains. For CRBN, three CVs form a tetrahedral geometry capturing the conical motion of the CRBN-CTD domain relative to other domains during the open-close transition.

### 3. Meta-eABF Simulation
Enhanced sampling is performed using meta-eABF (metadynamics with extended-Lagrangian adaptive biasing force):
- Fictitious CVs coupled to real CVs via harmonic springs
- Adaptive biasing force applied to fictitious CVs
- Metadynamics further enhances sampling of fictitious CV space
- Simulations run for 1 μs with frames saved every 200 ps

### 4. Free Energy Map Generation
Free energy landscapes are computed from meta-eABF simulations:
- CZAR (corrected Z-averaged restraints) algorithm estimates free energy gradients
- Poisson integration converts gradients to complete 3D free energy map
- Map spans CV space with millions of grid points

### 5. Informed Autoencoder Training
A dual-loss neural network is trained on MD trajectory data:
- **Encoder**: Compresses protein Cartesian coordinates to 3D latent space
- **Decoder**: Reconstructs full atomic coordinates from latent space
- **Loss1 (Latent Loss)**: Minimizes RMSD between latent space and CV coordinates
- **Loss2 (Reconstruction Loss)**: Minimizes RMSD between input and output structures
- Training uses Adam optimizer for 10,000 rounds with batch size 64

### 6. Structure Generation and Transition Prediction
Conformational transitions are predicted by:
- Defining anchor points in open, closed, and transition regions of FE map
- Fitting B-spline curves through low-energy pathways
- Generating CV coordinates along the path
- Using the informed decoder to predict full atomic structures
- Post-processing structures to fix geometric distortions

### 7. Validation and Quality Control
Generated structures undergo rigorous validation:
- Ramachandran plot analysis for backbone geometry
- Clash detection for steric conflicts
- Energy evaluation for physical plausibility
- RMSD comparison to reference structures

## Dependencies

### Core Requirements
- Python >= 3.8
- CUDA >= 11.0 (for GPU acceleration)

### Python Packages

**Deep Learning:**
```
tensorflow>=2.8.0 OR pytorch>=1.10.0
numpy>=1.21.0
scipy>=1.7.0
```

**Molecular Dynamics:**
```
openmm>=8.1.0
plumed>=2.9.0
mdanalysis>=2.0.0
```

**Structure Analysis:**
```
biopython>=1.79
prody>=2.0.0
```

**Visualization:**
```
matplotlib>=3.5.0
plotly>=5.0.0
imageio>=2.9.0
```

**Development:**
```
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
sphinx>=4.0.0
```

### System Requirements
- **GPU**: NVIDIA RTX 4080 or equivalent (minimum 12GB VRAM)
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 500GB+ for trajectory data and models
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, or macOS 11+

### Installation

```bash
# Clone repository
git clone https://github.com/[organization]/rmd.git
cd rmd

# Create conda environment
conda env create -f environment.yml
conda activate rmd

# Install package
pip install -e .
```

## Tests

### Running Tests

**Unit Tests:**
```bash
pytest tests/unit/
```

**Integration Tests:**
```bash
pytest tests/integration/
```

**Validation Tests (CRBN system):**
```bash
python scripts/validate_crbn.py
```

**Coverage Report:**
```bash
pytest --cov=rmd --cov-report=html tests/
```

### Test Organization
- `tests/unit/`: Unit tests for individual modules
- `tests/integration/`: End-to-end pipeline tests
- `tests/validation/`: Scientific validation and reproducibility tests

### Continuous Integration
All tests run automatically on push via GitHub Actions. Code coverage must be ≥80% for pull request approval.

## Usage

### Quick Start

```python
from rmd import SystemPreparation, MetaEABF, InformedAutoencoder

# 1. Prepare system
prep = SystemPreparation('structure.pdb')
system = prep.setup(force_field='ff14sb', water='tip3p')

# 2. Run meta-eABF simulation
sim = MetaEABF(system, cvs=['cv1', 'cv2', 'cv3'])
trajectory, fe_map = sim.run(duration='1us', save_interval='200ps')

# 3. Train informed autoencoder
model = InformedAutoencoder(latent_dim=3)
model.train(trajectory, fe_map, epochs=10000, batch_size=64)

# 4. Generate transition pathway
path = model.predict_transition(start='open', end='closed')
path.export('transition.pdb')
```

### Examples

See `examples/` directory for detailed tutorials:
- `01_crbn_reproduction.ipynb`: Reproduce CRBN open-close transition from paper
- `02_custom_system.ipynb`: Apply rMD to your own protein system
- `03_advanced_features.ipynb`: Data augmentation, reweighting, alternative CVs

### API Reference

Full API documentation available at: [https://rmd.readthedocs.io](https://rmd.readthedocs.io)

## Project Status

**Current Version:** 0.1.0-dev

**Development Phase:** Sprint 0 - Project Initialization

See `CHANGELOG.md` for version history and `PROJECT_PLAN.md` for detailed development roadmap.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{kolossvary2025rmd,
  title={Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process},
  author={Kolossv{\'a}ry, Istv{\'a}n and Coffey, Rory},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.02.12.638002}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines on:
- Code style (PEP 8 compliance required)
- Testing requirements (≥80% coverage)
- Documentation standards
- Pull request process

## Contact

**Project Maintainers:**
- [To be filled]

**Issues and Questions:**
- GitHub Issues: [https://github.com/[organization]/rmd/issues](https://github.com/[organization]/rmd/issues)

## Acknowledgments

This implementation is based on the research by István Kolossváry and Rory Coffey. We thank the original authors for their innovative work combining physics and machine learning for enhanced protein conformational sampling.

## Related Resources

- **Original Paper**: https://doi.org/10.1101/2025.02.12.638002
- **OpenMM Documentation**: http://openmm.org
- **Plumed Documentation**: https://www.plumed.org
- **TensorFlow/PyTorch**: https://www.tensorflow.org / https://pytorch.org
