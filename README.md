# Reinforced Molecular Dynamics (rMD) Project

This project aims to recreate the Reinforced Molecular Dynamics (rMD) technology as described in the research preprint "Reinforced molecular dynamics: Physics-infused generative machine learning model explores CRBN activation process" (doi: https://doi.org/10.1101/2025.02.12.638002).

## Agile Project Plan

### Sprint 1: Data Preprocessing and Basic Autoencoder

**Objective:** Establish foundational data handling and a basic autoencoder for protein structure compression.

*   **User Story 1.1: As a data scientist, I want to preprocess MD trajectory data so that it is in a suitable format for autoencoder training.**
    *   **Tasks:**
        *   **Task 1.1.1: Implement `flatten_coordinates` function.**
            *   **Description:** Create a function that flattens a 3D array of atomic coordinates `(num_atoms, 3)` into a 1D array `(num_atoms * 3,)`.
            *   **Acceptance Criteria:**
                *   Function correctly flattens a 2D NumPy array of shape (N, 3) to a 1D array of shape (N*3,).
                *   Handles edge cases like empty input or single atom input.
                *   Raises appropriate errors for invalid input types or shapes.
        *   **Task 1.1.2: Implement `superpose_structures` function.**
            *   **Description:** Create a function that takes a list/array of protein structures and a reference structure, superposing them to eliminate global rotational and translational degrees of freedom (e.g., using Kabsch algorithm principles).
            *   **Acceptance Criteria:**
                *   Function successfully superposes a set of structures to a reference structure.
                *   Returns structures where global rotation and translation relative to the reference are minimized.
                *   Handles multiple frames and ensures atom counts match.
                *   Raises appropriate errors for invalid input types or incompatible shapes.
        *   **Task 1.1.3: Create unit tests for data preprocessing functions.**
            *   **Description:** Develop comprehensive unit tests for `flatten_coordinates` and `superpose_structures`.
            *   **Acceptance Criteria:**
                *   All functions are tested with valid and invalid inputs.
                *   Tests cover various scenarios (e.g., different numbers of atoms, translated, rotated structures).
                *   Tests confirm correct output values and error handling.
                *   Code coverage for these functions is high.

*   **User Story 1.2: As a machine learning engineer, I want to build a basic autoencoder network so that I can compress and reconstruct protein structures.**
    *   **Tasks:**
        *   **Task 1.2.1: Implement the encoder component.**
            *   **Description:** Design and implement the encoder using fully connected layers with gradually decreasing neuron counts.
            *   **Acceptance Criteria:**
                *   Encoder accepts input of length 9696.
                *   Uses fully connected layers with "Swish" activation for hidden layers.
                *   Produces a latent space representation of 3 dimensions.
        *   **Task 1.2.2: Implement the decoder component.**
            *   **Description:** Design and implement the decoder using fully connected layers with gradually increasing neuron counts, mirroring the encoder.
            *   **Acceptance Criteria:**
                *   Decoder accepts input of 3 dimensions (latent space).
                *   Uses fully connected layers with "Swish" activation for hidden layers (and linear for output).
                *   Reconstructs output of length 9696.
        *   **Task 1.2.3: Integrate encoder and decoder into a full autoencoder model.**
            *   **Description:** Combine the encoder and decoder into a single TensorFlow `tf.keras.Model` or functional API model.
            *   **Acceptance Criteria:**
                *   The model takes an input of shape (None, 9696) and produces an output of shape (None, 9696).
                *   The latent space layer has 3 dimensions.
                *   The model can be compiled and summarized in TensorFlow.
        *   **Task 1.2.4: Create unit tests for the basic autoencoder.**
            *   **Description:** Develop unit tests to verify the autoencoder's architecture and forward pass.
            *   **Acceptance Criteria:**
                *   Tests confirm input and output shapes.
                *   Tests verify the latent space dimension.
                *   Tests ensure the model can perform a forward pass without errors.

### Sprint 2: Informed Autoencoder and Training Setup

**Objective:** Enhance the autoencoder with physical context and prepare the training pipeline.

*   **User Story 2.1: As a machine learning engineer, I want to modify the basic autoencoder into an informed autoencoder so that it incorporates physical context (free energy map).**
    *   **Tasks:**
        *   **Task 2.1.1: Implement the `latentLoss` function (Loss1).**
            *   **Description:** Create a loss function that calculates the "distance" or discrepancy between the latent space coordinates and the collective variable (CV) coordinates. (Paper mentions RMSD between point clouds).
            *   **Acceptance Criteria:**
                *   Function takes two sets of 3D coordinates (latent space, CVs) and returns a scalar loss value.
                *   Implementation reflects a suitable distance metric (e.g., RMSD or similar).
        *   **Task 2.1.2: Integrate Loss1 and Loss2 into a dual-loss training objective.**
            *   **Description:** Modify the autoencoder model or training loop to simultaneously optimize Loss1 (latent space to CVs) and Loss2 (reconstruction RMSD). This will likely involve a weighted sum of the two losses.
            *   **Acceptance Criteria:**
                *   Training process can compute and optimize both loss functions.
                *   Configurable weighting for Loss1 and Loss2.

*   **User Story 2.2: As a data scientist, I want to set up the data loading and preparation pipeline for training.**
    *   **Tasks:**
        *   **Task 2.2.1: Implement data loading for MD trajectories and CVs.**
            *   **Description:** Create functions to load MD trajectory frames (e.g., PDB, DCD, or similar) and corresponding 3D collective variable data. Assume data will be in a readily parsable format (e.g., NumPy arrays or CSV for CVs).
            *   **Acceptance Criteria:**
                *   Can load protein coordinates into `(num_frames, num_atoms, 3)` NumPy arrays.
                *   Can load CV coordinates into `(num_frames, 3)` NumPy arrays.
        *   **Task 2.2.2: Implement data batching and shuffling.**
            *   **Description:** Create a data generator or TensorFlow `tf.data` pipeline for efficient batching and shuffling of preprocessed MD trajectory and CV data.
            *   **Acceptance Criteria:**
                *   Provides batches of superposed and flattened protein structures and corresponding CVs.
                *   Data is shuffled before each epoch.
                *   Efficient memory usage.

*   **User Story 2.3: As a machine learning engineer, I want to configure the training loop and optimizer.**
    *   **Tasks:**
        *   **Task 2.3.1: Set up the training loop for the informed autoencoder.**
            *   **Description:** Implement the main training loop, including forward pass, dual-loss calculation, backpropagation, and parameter updates using the Adam optimizer.
            *   **Acceptance Criteria:**
                *   Training loop runs for a specified number of epochs.
                *   Updates model weights based on the combined loss.
        *   **Task 2.3.2: Implement model saving and loading.**
            *   **Description:** Add functionality to save trained model weights and load them for inference.
            *   **Acceptance Criteria:**
                *   Model weights can be saved after training.
                *   Saved weights can be loaded into an identical model architecture.

### Sprint 3: Evaluation, Prediction, and Refinement

**Objective:** Evaluate the model, implement structure generation, and refine outputs.

*   **User Story 3.1: As a scientist, I want to evaluate the performance of the trained rMD model.**
    *   **Tasks:**
        *   **Task 3.1.1: Calculate reconstruction RMSD (Loss2) on validation set.**
            *   **Description:** Implement a metric calculation to determine the average RMSD between input and reconstructed structures on a held-out validation set.
            *   **Acceptance Criteria:**
                *   Reports average Loss2 for the validation set.
                *   Calculation matches the definition in the paper.
        *   **Task 3.1.2: Visualize latent space mapping against free energy map.**
            *   **Description:** Develop a plotting script to visualize the trained latent space points and compare their distribution and coloring with the 3D free energy map. (This might be a manual comparison if FE map data is complex).
            *   **Acceptance Criteria:**
                *   Generates a 3D plot of latent space coordinates.
                *   Allows for coloring based on corresponding CVs or free energy values.

*   **User Story 3.2: As a scientist, I want to use the trained rMD model to generate new protein structures.**
    *   **Tasks:**
        *   **Task 3.2.1: Implement structure generation from latent space/FE map points.**
            *   **Description:** Create a function that takes 3D points (either from the latent space or directly from the free energy map in CV space) and uses the decoder to predict corresponding full atomistic protein structures.
            *   **Acceptance Criteria:**
                *   Takes 3D input coordinates and returns `(num_atoms, 3)` protein structures.
                *   Output structures are in the original (unflattened) format.
        *   **Task 3.2.2: Implement a path-following mechanism on the FE map for conformational transitions.**
            *   **Description:** Develop functionality to define a path in the CV/FE space (e.g., using B-splines as mentioned in the paper) and generate a series of structures along this path to simulate a conformational transition.
            *   **Acceptance Criteria:**
                *   Can take a sequence of 3D points defining a path.
                *   Generates a sequence of protein structures representing the transition.

*   **User Story 3.3: As a computational chemist, I want to post-process generated structures to fix local geometric distortions.**
    *   **Tasks:**
        *   **Task 3.3.1: Integrate a basic geometry cleanup step (placeholder).**
            *   **Description:** Implement a placeholder function for post-processing structures to relax local geometric distortions, acknowledging that full Rosetta Relax integration is beyond the scope of this initial package but a simple minimization or constraint might be useful. (Note: The paper mentions Rosetta Relax; a full integration might be a future enhancement. For now, a simple Python-based geometric check/correction or a placeholder for future extension is sufficient).
            *   **Acceptance Criteria:**
                *   A function `post_process_structure` exists that takes a protein structure and returns a refined one.
                *   Initial implementation can be a pass-through or a very basic geometric correction.
                *   Clear documentation that this is a placeholder for more advanced methods.

## Component & Dependency List

**Technology Stack:** Python

**Core Components:**

*   `rmd_data_preprocessing.py`: Module for functions to prepare MD trajectory data.
    *   `flatten_coordinates(coords)`: Flattens 3D coordinates to 1D.
    *   `superpose_structures(structures, reference_structure)`: Superposes multiple structures to a reference.
*   `rmd_autoencoder.py`: Module containing the autoencoder network definition.
    *   `rMDAutoencoder(input_dim, latent_dim, hidden_layers_encoder, hidden_layers_decoder)`: TensorFlow Keras Model for the autoencoder.
    *   `swish(x)`: Custom Swish activation function.
*   `rmd_losses.py` (Planned for Sprint 2): Module for custom loss functions.
    *   `reconstruction_loss(y_true, y_pred)`: RMSD-based reconstruction loss (Loss2).
    *   `latent_space_loss(latent_coords, cv_coords)`: Loss to align latent space with CVs (Loss1).
*   `rmd_training.py` (Planned for Sprint 2): Module for training utilities.
    *   `train_model(model, train_dataset, val_dataset, optimizer, epochs, ...)`: Training loop.
*   `rmd_prediction.py` (Planned for Sprint 3): Module for structure generation and analysis.
    *   `generate_structure_from_latent(model, latent_point)`: Generates structure from a latent point.
    *   `generate_transition_path(model, fe_map_path_points)`: Generates structures along a path in FE map.
    *   `post_process_structure(structure)`: Placeholder for geometric refinement.

**Required Libraries and Dependencies:**

*   **TensorFlow** (>= 2.x): For building and training the neural network models.
*   **NumPy**: For efficient numerical operations and array manipulation.
*   **SciPy**: Specifically `scipy.spatial.transform.Rotation` for robust structural superposition (Kabsch algorithm implementation).
*   **scikit-learn** (Potentially for future data manipulation/metrics, but not strictly required for initial autoencoder).
*   **Matplotlib/Seaborn** (For visualization, especially in Sprint 3).
*   **MDAnalysis** or similar library (Optional, for more complex parsing of MD trajectories if raw coordinate files are used, but we'll assume pre-parsed NumPy arrays for initial scope).

## Methodology Summary & Documentation Framework

### Methodology Summary

The Reinforced Molecular Dynamics (rMD) approach combines traditional Molecular Dynamics (MD) simulations with a novel physics-infused autoencoder neural network. The core idea is to train an autoencoder on MD trajectory data, but critically, to replace its generic latent space with a physically meaningful free-energy (FE) map derived from collective variables (CVs).

The process involves:

1.  **MD Simulations and FE Map Generation:** Running advanced sampling MD simulations (e.g., meta-eABF) to explore conformational space along defined collective variables and compute a 3D free energy map over these CVs.
2.  **Data Preprocessing:** Extracting protein structures from MD trajectories, superposing them to a common reference frame to remove global translation and rotation, and flattening their Cartesian coordinates into 1D vectors for neural network input.
3.  **Informed Autoencoder Design:**
    *   **Encoder:** Compresses the high-dimensional protein structure data into a low-dimensional latent space.
    *   **Decoder:** Reconstructs the original protein structure from the latent space representation.
    *   **Dual-Loss Training:** The autoencoder is trained with two simultaneous loss functions:
        *   `Loss2 (predLoss)`: Minimizes the reconstruction error (e.g., RMSD) between the input and reconstructed protein structures.
        *   `Loss1 (latentLoss)`: Minimizes the difference between the latent space coordinates and the physical collective variable (CV) coordinates for each trajectory frame. This is the "physics infusion" step, aligning the abstract latent space with the interpretable physical CV space.
4.  **Targeted Structure Generation:** Once trained, the free energy map (which now has an approximate one-to-one correspondence with the latent space) can be used to select points (e.g., low-energy regions, transition paths) from which the decoder can generate atomistic protein structures that are biologically relevant.
5.  **Post-processing:** Generated structures may undergo a cleanup step (e.g., local energy minimization) to fix geometric distortions inherent in ML-generated models.

This allows for efficient exploration of conformational space and the generation of meaningful protein structures and transition pathways, overcoming the physics-agnostic nature of traditional autoencoders.

### Documentation Framework

All project documentation will reside in Markdown files within the repository, with `README.md` serving as the master entry point.

## Introduction
*   High-level overview of the rMD project and its objectives.
*   Link to the scientific paper.

## Methodology
*   Detailed explanation of each implemented component, directly linking code to the scientific methodology described in the paper.
*   Includes descriptions of data preprocessing, autoencoder architecture, loss functions, training procedures, and structure generation.
*   References specific figures and sections of the paper where appropriate.

## Dependencies
*   Comprehensive list of all required Python libraries and their versions.
*   Instructions for setting up the development environment.

## Tests
*   Overview of the testing strategy (e.g., unit tests, integration tests).
*   Instructions on how to run tests.
*   Summary of test coverage.

### Code Quality and Standards

*   All Python code must adhere to **PEP 8** style guidelines.
*   All functions, classes, and complex code blocks must include clear, comprehensive **docstrings** explaining their purpose, arguments, and return values.
*   Code should be modular, readable, and efficient.
*   Version control (Git/GitHub) best practices will be followed, including meaningful commit messages and branching for features.

