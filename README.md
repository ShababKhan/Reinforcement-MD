# Reinforced Molecular Dynamics (rMD)

**Physics-Infused Generative Machine Learning for Protein Conformational Exploration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Introduction

This repository contains a Python implementation of **reinforced molecular dynamics (rMD)**, a novel computational method that combines molecular dynamics (MD) simulations with physics-infused machine learning to explore biologically relevant protein conformational transitions.

### Key Innovation

rMD replaces the arbitrary latent space of traditional autoencoders with a **physical free energy (FE) map** derived from enhanced sampling MD simulations. This enables:
- **Targeted structure generation** from low-energy regions
- **Conformational transition pathway exploration**
- **Desktop-scale computation** on a single GPU

### Scientific Application

We demonstrate rMD by modeling the conformational transition of **cereblon (CRBN)**, an E3 ligase substrate receptor crucial for molecular glue degrader therapies:
- **Open state** → **Closed state** transition upon IMiD binding
- 3D collective variable space capturing domain motions
- Full atomistic detail of transition mechanism

---

## Methodology

### 1. Enhanced Sampling MD Simulation

**Meta-eABF (Metadynamics + Extended Adaptive Biasing Force)**
- Biases simulation along 3 collective variables (CVs)
- CVs defined as distances between domain centers of mass
- Generates trajectory data for ML training

**System Details:**
- CRBN-DDB1 apo complex (~210,000 atoms)
- 2 × 1 μs simulations (open/closed starting conformations)
- OpenMM 8.1.0 + PLUMED 2.9.0
- 10,000 frames saved every 200 ps

### 2. Free Energy Map Construction

**CZAR Gradient Estimation:**
- Computes unbiased FE gradients on 3D grid
- Corrects for extended Lagrangian bias

**Poisson Integration:**
- Integrates gradients to obtain FE landscape
- 24M grid points spanning CV space
- 0-15 kcal/mol energy scale

### 3. Physics-Infused Autoencoder

**Network Architecture:**
- **Encoder:** Compresses 9,696D Cartesian coords → 3D latent space
- **Decoder:** Expands 3D latent space → 9,696D Cartesian coords
- Fully connected layers with Swish activation: `x·σ(x)`

**Dual-Loss Training:**
- **Loss1 (Latent Loss):** RMSD between latent space and CV coordinates
- **Loss2 (Reconstruction Loss):** MAE between input and output structures
- Weighted combination optimized via Adam

**Training Results:**
- Loss1 ≈ 1 Å (latent ↔ CV correspondence)
- Loss2 ≈ 1.6 Å (reconstruction accuracy)
- 10,000 rounds, batch size 64

### 4. Conformational Path Exploration

**B-Spline Path Definition:**
- Manually select anchor points on FE map
- Fit smooth B-spline through low-energy regions
- Generate interpolated points along path

**Structure Generation:**
- Decode CV coordinates to all-atom structures
- Post-process with Rosetta Relax + position-restrained minimization
- Export 20-frame transition trajectory

---

## Dependencies

### Python Environment
```bash
Python >= 3.8
```

### Core Libraries

**Molecular Dynamics:**
```
openmm >= 8.1.0
plumed >= 2.9.0
mdanalysis >= 2.0.0
biopython >= 1.79
```

**Machine Learning:**
```
tensorflow >= 2.8.0  # OR pytorch >= 1.10.0
scikit-learn >= 1.0.0
```

**Numerical Computing:**
```
numpy >= 1.21.0
scipy >= 1.7.0
pandas >= 1.3.0
```

**Visualization:**
```
matplotlib >= 3.4.0
seaborn >= 0.11.0
plotly >= 5.0.0
pymol-open-source >= 2.5.0
```

**Utilities:**
```
tqdm >= 4.62.0
h5py >= 3.6.0
pyyaml >= 6.0
pytest >= 7.0.0
```

### Installation

```bash
# Create conda environment
conda create -n rmd python=3.8
conda activate rmd

# Install dependencies
pip install -r requirements.txt

# Install OpenMM + PLUMED
conda install -c conda-forge openmm plumed
```

---

## Tests

### Unit Tests

**Run all tests:**
```bash
pytest tests/ -v
```

**Test coverage:**
```bash
pytest --cov=src tests/
```

### Scientific Validation Tests

1. **CV Calculation Accuracy**
   - Validate CV1, CV2, CV3 against known structures
   - Expected tolerance: ± 0.1 Å

2. **Encoder/Decoder Symmetry**
   - Input → Encode → Decode → Output
   - Expected RMSD: < 1.6 Å

3. **FE Map Gradient Integration**
   - Verify energy conservation
   - Check boundary conditions

4. **Structure Quality**
   - Ramachandran plot analysis
   - Clash detection
   - Hydrogen bond validation

---

## Usage

### Quick Start

```python
from src.autoencoder import rMDAutoencoder
from src.structure_generation import generate_transition_path

# Load trained model
model = rMDAutoencoder.load('models/rmd_crbn.h5')

# Define transition path on FE map
path_coords = generate_transition_path(
    start=[15.0, 20.0, 25.0],  # Open state CV coords
    end=[10.0, 15.0, 18.0],    # Closed state CV coords
    n_points=20
)

# Generate structures
structures = model.decode(path_coords)

# Save as PDB trajectory
save_trajectory(structures, 'output/transition.pdb')
```

### Full Pipeline

See `examples/crbn_analysis.ipynb` for complete workflow.

---

## Project Structure

```
rmd/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data_preparation.py      # PDB loading, superposition
│   ├── collective_variables.py  # CV definition
│   ├── md_simulation.py         # Meta-eABF wrapper
│   ├── free_energy.py           # FE map computation
│   ├── autoencoder.py           # Network architecture
│   ├── training.py              # Training loop
│   ├── structure_generation.py  # Path exploration
│   ├── postprocessing.py        # Geometric refinement
│   ├── visualization.py         # Figures and movies
│   └── utils.py                 # Helper functions
├── tests/
│   ├── test_cv.py
│   ├── test_autoencoder.py
│   ├── test_training.py
│   └── test_validation.py
├── examples/
│   └── crbn_analysis.ipynb
├── data/
│   ├── raw/                     # Input PDB files
│   ├── processed/               # Trajectories, FE maps
│   └── models/                  # Trained networks
├── docs/
│   ├── methodology.md
│   ├── api_reference.md
│   └── figures/
└── output/
    ├── structures/
    ├── figures/
    └── movies/
```

---

## Code Standards

- **PEP 8 compliance** enforced via `flake8`
- **Type hints** for all function signatures
- **Docstrings** in NumPy style
- **Unit tests** for all core functions
- **Git commit messages** follow conventional commits

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kolossvary2025rmd,
  title={Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process},
  author={Kolossv{\'a}ry, Istv{\'a}n and Coffey, Rory},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.02.12.638002}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Contact

**Project Manager:** AI Development Team  
**Repository:** [GitHub Link]

---

## Acknowledgments

Based on the research paper:
*Kolossváry & Coffey (2025). Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process. bioRxiv.*
