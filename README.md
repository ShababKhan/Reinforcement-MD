# Reinforced Molecular Dynamics (rMD) Software Recreation

## 1. Project Overview

This project aims to replicate the Reinforced Molecular Dynamics (rMD) methodology described in the paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" by István Kolossváry and Rory Coffey. rMD is a novel machine learning approach that integrates Molecular Dynamics (MD) simulation data with a physics-informed autoencoder to enable efficient exploration of protein conformational space and modeling of biologically relevant motions.

The core innovation is infusing physical context (specifically, free energy landscapes derived from collective variables) directly into the latent space of an autoencoder.

## 2. Methodology Recap

The rMD approach follows these key stages:

1.  **MD Simulation & Free Energy Map Generation:** Advanced sampling techniques (e.g., metadynamics, meta-eABF) are used with MD simulations to bias sampling along relevant Collective Variables (CVs). This process generates a 3D free energy (FE) map representing the conformational landscape.
2.  **Autoencoder Architecture:** A standard encoder-decoder neural network is employed to compress protein structure representations into a low-dimensional latent space (LS) and reconstruct them.
3.  **Physics Infusion (rMD Core):** A dual-loss training strategy is implemented:
    *   **$\text{Loss}_2$ ($\text{predLoss}$):** Minimizes the reconstruction error (RMSD) between input and output structures.
    *   **$\text{Loss}_1$ ($\text{latentLoss}$):** Minimizes the error between the latent space representation and the target CV coordinates. This forces the latent space to become a physically meaningful representation of the CV space.
4.  **Dual-Loss Training:** The network is trained simultaneously to optimize both $\text{Loss}_1$ and $\text{Loss}_2$, effectively infusing physical information into the latent representation. A weighting factor ($\lambda$) balances these two objectives.
5.  **Structure Generation & Exploration:** The trained informed autoencoder, guided by the FE map (or directly by CV coordinates), can generate novel protein conformations, particularly in low-free-energy regions, and explore transition pathways.
6.  **Post-processing:** Generated structures may undergo refinement (e.g., using Rosetta Relax) to correct local geometric distortions.

## 3. Implemented Components

### 3.1. Basic Autoencoder (`autoencoder_basic.py`)

*   **`CRBNAutoencoder`:** Implements the core encoder-decoder architecture with Swish activation, mapping input structure features to a latent space and reconstructing them.
*   **`calculate_rmsd_loss`:** Defines $\text{Loss}_2$ (predictive loss), calculating the Mean Squared Error (MSE) between input and reconstructed structures. This is proportional to RMSD.

### 3.2. Basic Autoencoder Training (`train_autoencoder.py`)

*   **Dummy Data Generation:** Creates synthetic structural data matching the expected input dimensions (e.g., 9696 features).
*   **`Dataset` & `DataLoader`:** PyTorch utilities for batching dummy data.
*   **Training Loop:** Trains the `CRBNAutoencoder` using Adam optimizer and only $\text{Loss}_2$ for a set number of epochs.
*   **Model Persistence:** Saves the trained basic autoencoder weights to `basic_autoencoder_model.pth`.

### 3.3. Informed Autoencoder Utilities (`informed_autoencoder_utils.py`)

*   **`calculate_cv_loss`:** Defines $\text{Loss}_1$ (latent-CV alignment loss), calculating MSE between latent space vectors and target CV coordinates.
*   **`combined_loss`:** Function to calculate the weighted sum of $\text{Loss}_1$ and $\text{Loss}_2$.

### 3.4. Informed Autoencoder Training (`train_informed_autoencoder.py`)

*   **Dummy CV Target Generation:** Creates synthetic CV target data correlated with underlying "ideal" latent representations.
*   **Loading Basic AE Weights:** Initializes the informed autoencoder by loading weights from the pre-trained basic model.
*   **Dual-Loss Training Loop:** Trains the model using both $\text{Loss}_1$ and $\text{Loss}_2$ (via `combined_loss`) to infuse physical context.
*   **Model Persistence:** Saves the weights of the trained informed autoencoder to `informed_autoencoder_model.pth`.

### 3.5. Unit Tests (`test_*.py`)

*   `test_autoencoder_basic.py`: Verifies the structure and `predLoss` calculation of the basic AE.
*   `test_train_autoencoder.py`: Tests the dummy data generation, DataLoader, and the training execution/saving mechanism.

### 3.6. Data Handling, CV Extraction & FE Map Integration (Sprint 2 - Focus)

This development phase focuses on establishing the capability to process real Molecular Dynamics (MD) simulation data, extract relevant Collective Variables (CVs), and handle Free Energy (FE) maps, which are foundational inputs for the rMD approach.

*   **Simulated Data Processing Modules:**
    *   **`data_processing/trajectory_loader.py`:** (Conceptual - to be implemented) This module will be responsible for reading MD trajectory files (e.g., PDB, DCD, XTC) and extracting atomic coordinates. It will leverage libraries like `MDAnalysis` for robust parsing and will include functionality for essential preprocessing steps such as aligning structures to a reference frame to remove global rotations and translations, as noted in the paper.
    *   **`data_processing/cv_calculator.py`:** (Conceptual - to be implemented) This module will implement functions to calculate the specific Collective Variables (CVs) described in the paper. This involves identifying key protein domains (e.g., CRBN-NTD, CRBN-CTD), calculating their Centers of Mass (COM), and determining the distances/orientations that constitute the 3D CV space (as visualized in Figure S1).
    *   **`data_processing/fe_map_handler.py`:** (Conceptual - to be implemented) This module will provide utilities for loading or approximating Free Energy (FE) maps. Given that generating FE maps requires substantial simulation resources, the initial implementation will likely focus on reading pre-computed FE map data from files (e.g., `.npy` grids). If direct FE data is unavailable, mechanisms to use the *distribution* of generated CVs as a proxy during training will be explored.

*   **Integration into Training Data Pipeline:** These new data handling modules will be integrated into the existing PyTorch `Dataset` and `DataLoader` infrastructure. This ensures that the informed autoencoder training script (`train_informed_autoencoder.py`) can ingest data derived from actual (or simulated) MD trajectories and their corresponding CVs/FE maps.

*   **Addressing `requirements.txt` Issue:**
    We have encountered a persistent issue with reliably creating the `requirements.txt` file using the available tooling. For now, please manually ensure the dependencies listed in Section 5 (Installation) are installed in your environment. We will revisit this tool functionality.

### Current Status

The foundational autoencoder architecture, its training scripts, and utility functions for physics-informed loss calculation are complete and documented. The project is now poised to integrate with real MD simulation data, CV extraction, and FE map handling, marking the next significant step in replicating the rMD methodology.

## 4. Dependencies

The project relies on Python and several key libraries. Please refer to `requirements.txt` for specific version requirements.

*   **Core ML:** PyTorch
*   **Numerical Computations:** NumPy
*   **Testing:** Pytest

## 5. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rmd_replication
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` creation fails, manually install packages listed in Section 4 based on your environment.)*

## 6. Usage

All training scripts assume the necessary `autoencoder_basic.py` and `informed_autoencoder_utils.py` are in the same directory or accessible via Python path.

### 6.1. Training the Basic Autoencoder

This trains the autoencoder using only the reconstruction loss ($\text{Loss}_2$).

```bash
python train_autoencoder.py
```
This will generate `basic_autoencoder_model.pth`.

### 6.2. Training the Informed Autoencoder

This trains the autoencoder using both $\text{Loss}_1$ and $\text{Loss}_2$ from the pre-trained basic autoencoder.

```bash
python train_informed_autoencoder.py
```
This will generate `informed_autoencoder_model.pth`.

*Note: The `LAMBDA_LOSS1` parameter in `train_informed_autoencoder.py` controls the balance between reconstruction and physical alignment. Its value may need tuning.*

### 6.3. Running Unit Tests

To ensure all core components function as expected:

```bash
pytest
```

## 7. File Structure

```
.
├── autoencoder_basic.py
├── informed_autoencoder_utils.py
├── train_autoencoder.py
├── train_informed_autoencoder.py
├── test_autoencoder_basic.py
├── test_train_autoencoder.py
├── README.md             # This file
├── requirements.txt
# Trained models will be generated locally:
# ├── basic_autoencoder_model.pth
# └── informed_autoencoder_model.pth
# Data processing modules (to be implemented):
# ├── data_processing/
# │   ├── __init__.py
# │   ├── trajectory_loader.py
# │   ├── cv_calculator.py
# │   └── fe_map_handler.py
```

## 8. Future Work

*   Implement Collective Variable (CV) calculation and FE map generation from actual MD trajectories.
*   Integrate real protein structure data (e.g., CRBN PDB files and trajectory frames).
*   Develop functionality for generating structures along user-defined paths in CV space and post-processing/visualization (as per Figure 4 and supplemental materials).
*   Investigate and implement alternative loss functions mentioned in the paper.

## 9. License

[Specify your chosen license here, e.g., MIT License]
