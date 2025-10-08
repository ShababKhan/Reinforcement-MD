# rMD: Reinforced Molecular Dynamics

## Project Overview

This project aims to recreate the "Reinforced molecular dynamics (rMD)" scientific software based on the research paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" by István Kolossváry and Rory Coffey (doi: https://doi.org/10.1101/2025.02.12.638002).

The core idea of rMD is to combine molecular dynamics (MD) trajectory data and free-energy (FE) map data to train a dual-loss function autoencoder network. This "informed autoencoder" can explore conformational space more efficiently than traditional MD simulations by infusing the latent space with a physical context derived from the FE map. This enables targeted protein structure prediction and the exploration of biologically relevant protein motions, such as the open-to-closed conformational transition of CRBN.

## Methodology

### Sprint 1: Data Simulation & Core Autoencoder (Basic)

#### Task 1.1: Generate Synthetic/Mock Input Data
- **Description:** Implemented `generate_mock_data.py` to create synthetic `input_coords` (representing flattened Cartesian coordinates of protein structures, shape: 10,000x9696) and `cv_coords` (representing 3-dimensional collective variable data, shape: 10,000x3). This script generates random float values between 0 and 1 for both datasets, serving as mock inputs for training the autoencoder. This directly corresponds to the necessity for input data from MD simulations and associated collective variables, as described in the paper's "Informed autoencoder network" section and Figure S2, before the autoencoder training can commence.

#### Task 1.2: Implement `rmsd_loss` Function (Loss2)
- **Description:** Developed `src/losses/rmsd_loss.py`, which contains the `rmsd_loss` function. This PyTorch-based function calculates a Root Mean Square Deviation (RMSD)-like metric between two sets of flattened Cartesian coordinates. It serves as the `Loss2` function for the autoencoder, minimizing the structural superposition error between input and reconstructed protein structures, as detailed in the paper's "Informed autoencoder network" section and Figure 2. The implementation assumes pre-aligned structures, consistent with the paper's methodology of superposing trajectory frames prior to network training.

## Project Structure

```
.
├── README.md
├── generate_mock_data.py
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── autoencoder.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── rmsd_loss.py
│   └── trainer/
│       ├── __init__.py
│       └── basic_trainer.py
└── tests/
    ├── __init__.py
    └── test_generate_mock_data.py
```

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ShababKhan/Reinforcement-MD.git
    cd Reinforcement-MD
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS
    source venv/bin/activate
    # On Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: `requirements.txt` will be created as we identify dependencies.)

## Usage

Instructions on how to run the simulations, train models, and generate structures will be added here as features are implemented.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes, ensuring adherence to coding standards (PEP 8).
4.  Write comprehensive unit tests for your code.
5.  Update documentation (docstrings, `README.md`).
6.  Submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or inquiries, please open an issue in the GitHub repository.
