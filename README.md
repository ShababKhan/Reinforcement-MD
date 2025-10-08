# rMD (Reinforced Molecular Dynamics) Project Blueprint

## Project Overview

This project aims to recreate the Reinforced Molecular Dynamics (rMD) scientific software based on the research paper "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" by István Kolossváry and Rory Coffey (doi: https://doi.org/10.1101/2025.02.12.638002). The rMD technology combines MD trajectory data and free-energy (FE) map data to train a dual-loss function autoencoder network capable of exploring conformational space more efficiently than traditional MD simulations.

## Methodology Summary

The core methodology involves:
1.  **Data Preparation:** Generating or processing molecular dynamics (MD) simulation trajectory data and associated free-energy (FE) map data.
2.  **Autoencoder Design:** Implementing an "informed autoencoder" network. This network differs from a basic autoencoder by replacing the latent space with a physical context, specifically a free-energy (FE) map, which is computed from MD simulations over low-dimensional collective variables (CVs).
3.  **Dual-Loss Function Training:** The autoencoder is trained using a dual-loss function:
    *   **Loss1 ("latentLoss"):** Minimizes the difference between the latent space coordinates and the collective variable (CV) coordinates, effectively linking the latent space to the physical CV space.
    *   **Loss2 ("predLoss"):** Minimizes the Root Mean Square Distance (RMSD) between the input protein structures and the autoencoder's reconstructed output structures, ensuring accurate structural regeneration.
4.  **Conformational Space Exploration:** Utilizing the trained autoencoder and the FE map to:
    *   Generate new protein structures in poorly sampled regions.
    *   Explore conformational transitions by following paths on the FE map.
5.  **Post-processing:** Refining generated structures to relax local geometric distortions.

## Technology Stack

*   **Python:** Primary programming language.
*   **PyTorch:** For building and training the neural network models.
*   **NumPy:** For numerical operations and data handling.
*   **SciPy:** Potentially for advanced scientific computing tasks if needed (e.g., spatial calculations for RMSD).
*   **Matplotlib/Seaborn:** For data visualization (e.g., FE maps, conformational paths).
*   **Pytest:** For unit testing.

## Project Structure

```
rMD_Project/
├── src/
│   ├── generate_mock_data.py
│   ├── losses.py
│   ├── models.py
│   └── trainer.py
├── tests/
│   ├── test_generate_mock_data.py
│   └── test_losses.py
├── docs/
│   └── README.md (This file)
└── requirements.txt
```

## Agile Project Plan - Sprint 1: Data Simulation & Core Autoencoder (Basic)

**Sprint Goal:** Establish foundational data generation and the basic autoencoder components, ensuring they are testable and documented.

---

### **Task 1.1: Generate Synthetic/Mock Input Data**

*   **Objective:** Create a Python script (`src/generate_mock_data.py`) to produce synthetic `input_coords` (10,000x9696 float array) and `cv_coords` (10,000x3 float array) to simulate protein trajectory data and collective variable data. This mock data will be used for initial development and testing of the autoencoder.
*   **Acceptance Criteria:**
    *   `src/generate_mock_data.py` exists and contains the `generate_mock_data` function.
    *   The script runs without errors and outputs two NumPy arrays: `input_coords` with shape (10000, 9696) and `cv_coords` with shape (10000, 3).
    *   Output arrays are of `np.float32` dtype.
    *   Output arrays contain reasonable numerical values (e.g., `input_coords` values between -100 and 100, `cv_coords` values between 0 and 50).
    *   Comprehensive docstrings are included in the script.
    *   Corresponding unit tests are implemented in `tests/test_generate_mock_data.py` covering shapes, types, and ranges.
    *   A summary is added to the "Methodology" section of `README.md` linking this function to the paper's data generation/preparation aspects.
*   **Status:** **COMPLETED**
    *   `src/generate_mock_data.py` created and updated.
    *   Docstrings added.
    *   `tests/test_generate_mock_data.py` created with tests for shapes, types, and ranges.
    *   `README.md` updated to reflect this completion.

---

### **Task 1.2: Implement `rmsd_loss` Function (Loss2)**

*   **Objective:** Develop a Python function (`src/losses.py:rmsd_loss`) that calculates a metric equivalent to the Root Mean Square Distance (RMSD) between two sets of flattened Cartesian coordinates (`y_pred` and `y_true`). This will serve as Loss2 ("predLoss") for the autoencoder, minimizing the structural superposition error.
*   **Acceptance Criteria:**
    *   `src/losses.py` exists and contains the `rmsd_loss` function.
    *   The `rmsd_loss` function takes two PyTorch tensors of the same shape as input.
    *   It correctly calculates an RMSD-like value (square root of the mean squared difference over all elements).
    *   It handles edge cases gracefully (e.g., identical inputs result in 0 loss, different inputs result in non-zero loss).
    *   It raises an appropriate error (e.g., `ValueError`) if input tensor shapes do not match.
    *   Comprehensive docstrings are included in the function.
    *   Corresponding unit tests are implemented in `tests/test_losses.py` covering identical inputs, different inputs, shape mismatches, and batch processing.
    *   A summary is added to the "Methodology" section of `README.md` linking this function to the paper's Loss2 definition.
*   **Status:** **COMPLETED**
    *   `src/losses.py` created and updated.
    *   Docstrings added.
    *   `tests/test_losses.py` created with tests for various scenarios.
    *   `README.md` updated to reflect this completion.

---

### **Task 1.3: Design and Implement Basic Autoencoder Architecture**

*   **Objective:** Implement the `BasicAutoencoder` class in `src/models.py` using PyTorch. This autoencoder will comprise fully connected (Linear) layers with specific dimensions: 9696 -> 5000 -> 1000 -> 500 -> 3 (latent space) -> 500 -> 1000 -> 5000 -> 9696. The Swish activation function (x * sigmoid(x)) should be applied after each hidden layer, except for the latent space and the final output layer.
*   **Acceptance Criteria:**
    *   `src/models.py` exists and contains the `BasicAutoencoder` class, inheriting from `torch.nn.Module`.
    *   The `__init__` method correctly defines all specified Linear layers and Swish activation functions.
    *   The `forward` method correctly implements the encoder and decoder pass, ensuring data flows through the specified layers and activations, returning the reconstructed output and the latent space representation.
    *   Comprehensive docstrings are included for the class and its methods.
    *   Corresponding unit tests are implemented in `tests/test_models.py` to verify the architecture, layer dimensions, and forward pass functionality.
    *   A summary is added to the "Methodology" section of `README.md` linking this class to the paper's autoencoder design.
*   **Status:** PENDING

---

### **Task 1.4: Implement Training Loop for Basic Autoencoder**

*   **Objective:** Create a Python function (`src/trainer.py:train_basic_autoencoder`) to train the `BasicAutoencoder` using the mock data generated in Task 1.1 and the `rmsd_loss` (Loss2) from Task 1.2. The training loop should use the Adam optimizer, with a batch size of 64 and run for 10,000 epochs.
*   **Acceptance Criteria:**
    *   `src/trainer.py` exists and contains the `train_basic_autoencoder` function.
    *   The function takes the autoencoder model, mock input data, mock CV data (though only input data used for basic AE), optimizer, and number of epochs as arguments.
    *   The training loop correctly iterates through epochs and batches.
    *   It performs a forward pass, calculates `rmsd_loss`, performs backpropagation, and updates model weights using the Adam optimizer.
    *   The `rmsd_loss` decreases over epochs (demonstrating learning).
    *   Comprehensive docstrings are included in the function.
    *   Corresponding unit tests are implemented in `tests/test_trainer.py` to verify the training loop's functionality, ensuring loss reduction.
    *   A summary is added to the "Methodology" section of `README.md` linking this function to the paper's training process.
*   **Status:** PENDING

---

## Documentation Standards

*   All Python modules and functions must include comprehensive docstrings following a consistent style (e.g., Google or Sphinx style).
*   The `README.md` will serve as the master documentation, updated incrementally with summaries of completed tasks, methodology links, and relevant acceptance criteria.

## Testing Standards

*   Unit tests will be written using `pytest`.
*   Tests should cover core functionality, edge cases, and adherence to acceptance criteria.
*   All new code must have corresponding passing unit tests before being considered complete.
