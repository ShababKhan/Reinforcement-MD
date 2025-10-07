# Reinforced Molecular Dynamics (rMD)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Physics-infused generative machine learning model for exploring protein conformational space**

This software is a recreation of the Reinforced Molecular Dynamics (rMD) technology described in:

> Kolossváry, I., & Coffey, R. (2025). Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process. *bioRxiv*. https://doi.org/10.1101/2025.02.12.638002

## Overview

Reinforced Molecular Dynamics (rMD) is a novel computational method that combines molecular dynamics (MD) simulation data with machine learning to efficiently explore protein conformational space. The key innovation is a **dual-loss autoencoder** that replaces the abstract latent space with a **physics-based free energy (FE) map**, enabling targeted protein structure generation and conformational transition modeling.

### Key Features

- 🧬 **Physics-informed ML**: Latent space directly corresponds to free energy landscape
- 🖥️ **Desktop-scale**: Runs on a single GPU workstation (no supercomputer required)
- 🔬 **Self-contained**: No pre-trained models needed; trained on your own MD data
- ⚡ **Efficient exploration**: Generate structures in poorly sampled regions
- 🛤️ **Transition paths**: Model conformational changes along low-energy pathways
- 📊 **Comprehensive visualization**: Interactive 3D plots, animations, and analysis tools

## Scientific Background

Traditional protein structure prediction methods (AlphaFold, RosettaFold) excel at predicting static structures but cannot directly model protein dynamics. rMD bridges this gap by:

1. **Defining collective variables (CVs)** that capture biologically relevant motions
2. **Computing free energy maps** from advanced MD simulations (meta-eABF)
3. **Training an informed autoencoder** that links the latent space to the FE map
4. **Generating structures** by sampling the FE map and using the decoder

The result is a powerful tool for understanding protein conformational transitions critical to biological function, drug discovery, and molecular glue degrader development.

## Installation

### System Requirements

**Hardware:**
- GPU: NVIDIA RTX 4080 or equivalent (12+ GB VRAM recommended)
- CPU: 8+ cores, 3+ GHz
- RAM: 32+ GB
- Storage: 100+ GB SSD

**Software:**
- Operating System: Linux (Ubuntu 22.04+), macOS 11+, or Windows 10+ with WSL2
- CUDA: 11.8+ (for GPU acceleration)
- Python: 3.8-3.11

### Quick Install

```bash
# Clone the repository
git clone https://github.com/ShababKhan/Reinforcement-MD.git
cd Reinforcement-MD

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate rmd

# Verify installation
python -c "import rmd; print(rmd.__version__)"
```

For detailed installation instructions, see [docs/installation.md](docs/installation.md).

## Quick Start

### 1. Prepare Your Data

```python
from rmd.data import TrajectoryLoader, StructureProcessor

# Load MD trajectory
loader = TrajectoryLoader("path/to/trajectory.dcd", "path/to/topology.pdb")
structures = loader.load_frames(every=200)  # Load every 200th frame

# Superpose structures
processor = StructureProcessor(structures)
aligned_coords = processor.superpose_all(reference=0)
```

### 2. Compute Collective Variables

```python
from rmd.data import CollectiveVariables

# Define CVs for your system
cv_calculator = CollectiveVariables(
    domain_residues={
        "NTD": range(1, 100),
        "CTD": range(100, 200),
        "HBD": range(200, 250),
        "BPC": range(250, 400),
    }
)

# Compute CV coordinates for all frames
cv_coords = cv_calculator.compute_all(structures)
```

### 3. Train the rMD Autoencoder

```python
from rmd.network import InformedAutoencoder
from rmd.training import Trainer

# Initialize network
model = InformedAutoencoder(
    input_dim=9696,  # Number of heavy atoms * 3
    latent_dim=3,
    hidden_dims=[4096, 2048, 1024, 512, 256, 128, 64],
)

# Train model
trainer = Trainer(
    model=model,
    data=aligned_coords,
    cv_coords=cv_coords,
    batch_size=64,
    epochs=10000,
    device="cuda",
)

history = trainer.train()
```

### 4. Generate Structures from Free Energy Map

```python
from rmd.generation import StructureGenerator
from rmd.data import FreeEnergyMap

# Load FE map
fe_map = FreeEnergyMap.from_file("path/to/fe_map.dat")

# Sample low-energy regions
low_energy_cvs = fe_map.sample_low_energy(n_samples=100, threshold=5.0)  # < 5 kcal/mol

# Generate structures
generator = StructureGenerator(model)
structures = generator.from_cv_coords(low_energy_cvs)

# Export to PDB
generator.export_structures(structures, "output_dir/")
```

### 5. Compute Conformational Transition Path

```python
from rmd.generation import PathComputer

# Define start and end points (e.g., open and closed states)
start_cv = [25.0, 30.0, 35.0]  # Open state
end_cv = [15.0, 18.0, 20.0]    # Closed state

# Compute path on FE map
path_computer = PathComputer(fe_map)
path_cvs = path_computer.compute_path(
    start=start_cv,
    end=end_cv,
    n_points=20,
    method="bspline",
)

# Generate structures along path
transition_structures = generator.from_cv_coords(path_cvs)

# Create animation
from rmd.visualization import Animation
anim = Animation(transition_structures)
anim.create_movie("transition.mp4", fps=5)
```

## Documentation

- **[Project Blueprint](docs/PROJECT_BLUEPRINT.md)**: Complete project plan and specifications
- **[Methodology](docs/methodology.md)**: Detailed scientific methodology
- **[Installation Guide](docs/installation.md)**: Comprehensive setup instructions
- **[User Tutorial](docs/tutorials/)**: Step-by-step guides
- **[API Reference](docs/api/)**: Complete API documentation
- **[Examples](examples/)**: Jupyter notebooks and scripts

## Project Status

🚧 **Currently in Development** 🚧

- [x] Project blueprint complete
- [ ] Sprint 1: Foundation & Data Infrastructure (Weeks 1-2)
- [ ] Sprint 2: Autoencoder Architecture (Weeks 3-4)
- [ ] Sprint 3: Training Pipeline (Weeks 5-6)
- [ ] Sprint 4: FE Map Integration (Weeks 7-8)
- [ ] Sprint 5: Post-Processing (Weeks 9-10)
- [ ] Sprint 6: Visualization (Weeks 11-12)
- [ ] Sprint 7: Testing & Documentation (Weeks 13-14)
- [ ] Sprint 8: Advanced Features (Weeks 15-16)

**MVP Target**: End of Sprint 7 (14 weeks)  
**Full Release**: End of Sprint 8 (16 weeks)

See [docs/PROJECT_BLUEPRINT.md](docs/PROJECT_BLUEPRINT.md) for the complete development plan.

## Examples

### CRBN Open-Closed Transition

The paper demonstrates rMD on the conformational transition of cereblon (CRBN), an E3 ligase substrate receptor critical for molecular glue degrader function:

```python
# See examples/notebooks/crbn_transition.ipynb for full example
from rmd.examples import CRBNExample

example = CRBNExample()
example.run_complete_workflow()
```

This reproduces Figures 3 and 4 from the paper, showing:
- Latent space vs. free energy map correspondence
- Open-to-closed transition path
- Generated structures along the transition

## Citation

If you use this software in your research, please cite:

```bibtex
@article{kolossvary2025reinforced,
  title={Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process},
  author={Kolossv{\'a}ry, Istv{\'a}n and Coffey, Rory},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.02.12.638002}
}
```

And this software implementation:

```bibtex
@software{rmd_python,
  title={Reinforced Molecular Dynamics (rMD) - Python Implementation},
  author={[Your Name]},
  year={2025},
  url={https://github.com/ShababKhan/Reinforcement-MD}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/ShababKhan/Reinforcement-MD.git
cd Reinforcement-MD
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=rmd

# Check code style
black rmd/ tests/
flake8 rmd/ tests/
mypy rmd/
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original rMD methodology: István Kolossváry and Rory Coffey
- Inspired by the degiacomi autoencoder design (Degiacomi, 2019)
- Built with PyTorch, MDAnalysis, OpenMM, and other open-source tools

## Contact

- **Project Lead**: [Your Name]
- **Issues**: https://github.com/ShababKhan/Reinforcement-MD/issues
- **Discussions**: https://github.com/ShababKhan/Reinforcement-MD/discussions

## Support

If you encounter any problems or have questions:

1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/ShababKhan/Reinforcement-MD/issues)
3. Create a [new issue](https://github.com/ShababKhan/Reinforcement-MD/issues/new) with:
   - System information
   - Error messages
   - Minimal reproducible example

---

**Made with ❤️ for the computational biology community**
