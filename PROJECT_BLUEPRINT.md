# Reinforced Molecular Dynamics (rMD) Project Blueprint

## Overview
This project implements the Reinforced Molecular Dynamics (rMD) algorithm described in the paper by István Kolossváry and Rory Coffey, which combines molecular dynamics with machine learning to explore conformational transitions of proteins. The key innovation of rMD is replacing the latent space of an autoencoder with a physical free energy map, allowing for targeted exploration of biologically relevant protein motions.

## 1. Agile Project Plan

### Sprint 1: Project Setup and System Design (Week 1)
**User Stories:**
- As a developer, I need a proper project structure with environment configurations to begin development
- As a developer, I need to understand the overall system architecture and workflow
- As a researcher, I need to collect and prepare test datasets for the CRBN protein system

**Tasks:**
- [x] Create repository structure and documentation framework
- [ ] Define core classes and interfaces for the system
- [ ] Set up development environment with dependencies
- [ ] Create data processing pipeline
- [ ] Obtain and process PDB structures for CRBN (6H0G - closed form, 6H0F - open form)

**Acceptance Criteria:**
- Repository is properly structured with documentation
- Development environment can be easily recreated
- Initial UML diagrams and architecture documents are complete
- CRBN protein structures are processed and ready for simulation

### Sprint 2: Collective Variables and Molecular Dynamics Setup (Week 2)
**User Stories:**
- As a researcher, I need to define proper collective variables (CVs) for CRBN open-close conformational transitions
- As a developer, I need to create interfaces for MD simulation data and free energy maps
- As a user, I need tools to process and analyze MD trajectories

**Tasks:**
- [ ] Implement collective variable definitions for CRBN (COM distances as described in paper)
- [ ] Create interfaces for OpenMM/Plumed integration
- [ ] Implement trajectory processing utilities
- [ ] Design free energy map calculation and storage

**Acceptance Criteria:**
- CV calculations correctly identify CRBN open vs closed states
- MD simulation interface can handle trajectory data
- Free energy map representation is defined
- Test cases verify CV calculations on example structures

### Sprint 3: Autoencoder Implementation (Week 3)
**User Stories:**
- As a researcher, I need a customizable autoencoder architecture for protein structure prediction
- As a developer, I need efficient data handling for training and validation
- As a user, I need to visualize the latent space and model performance

**Tasks:**
- [ ] Implement encoder architecture with proper layers
- [ ] Implement decoder architecture with reconstruction capabilities
- [ ] Create training pipeline with data loading and preprocessing
- [ ] Implement basic loss functions (Loss2/predLoss)
- [ ] Design visualization tools for latent space

**Acceptance Criteria:**
- Autoencoder can compress and reconstruct protein structures
- Training pipeline handles batch processing and validation
- Average RMSD of reconstruction is tracked during training
- Latent space visualization shows meaningful clusters

### Sprint 4: Physics-Infused Learning (Week 4)
**User Stories:**
- As a researcher, I need to link the latent space with the physical CV space
- As a developer, I need to implement the dual loss function training
- As a user, I need tools to evaluate model performance

**Tasks:**
- [ ] Implement CV-based loss function (Loss1/latentLoss)
- [ ] Create combined loss function for dual training
- [ ] Implement model validation and performance metrics
- [ ] Optimize hyperparameters for dual-loss training

**Acceptance Criteria:**
- Latent space shows clear mapping to CV coordinates
- Training converges with meaningful dual loss metrics
- Model can reproduce protein structures with RMSD < 2Å
- Latent space distribution matches the free energy map shape

### Sprint 5: Path Finding and Structure Generation (Week 5)
**User Stories:**
- As a researcher, I need to generate transition paths between open and closed CRBN states
- As a developer, I need to create tools for structure generation from CV points
- As a user, I need to visualize and analyze transition pathways

**Tasks:**
- [ ] Implement path-finding algorithms on free energy maps
- [ ] Create structure generation from CV coordinates
- [ ] Implement B-spline interpolation for paths
- [ ] Add structural post-processing for generated structures

**Acceptance Criteria:**
- System can generate a transition path between open and closed CRBN states
- Generated structures have reasonable geometry and minimal distortion
- Transition path follows low-energy regions in the free energy landscape
- Visualizations clearly demonstrate the conformational transition

### Sprint 6: Testing, Documentation, and Packaging (Week 6)
**User Stories:**
- As a user, I need comprehensive documentation to use the system
- As a developer, I need proper tests to ensure system reliability
- As a researcher, I need example workflows for different applications

**Tasks:**
- [ ] Create comprehensive test suite
- [ ] Complete API documentation
- [ ] Create usage examples and tutorials
- [ ] Package software for distribution

**Acceptance Criteria:**
- All core components have unit tests with good coverage
- Documentation includes theoretical background and practical examples
- Example workflows demonstrate key capabilities
- Software can be installed via standard methods (pip, conda)

## 2. Component & Dependency List

### Core Components:
1. **Molecular Dynamics Interface**
   - Trajectory data management
   - CV calculation
   - Free energy map integration

2. **Data Processing Pipeline**
   - Protein structure preprocessing
   - Trajectory frame extraction
   - Data augmentation

3. **Autoencoder Model**
   - Encoder network
   - Decoder network
   - Dual-loss training system

4. **Path Finding and Structure Generation**
   - Free energy map navigation
   - B-spline interpolation
   - Structure generation and refinement
   - Visualization tools

### Dependencies:
- **Python 3.9+**: Core language
- **NumPy**: Numerical operations
- **SciPy**: Scientific computing, optimization, B-spline interpolation
- **PyTorch**: Neural network framework
- **MDAnalysis**: Molecular dynamics trajectory analysis
- **OpenMM**: Molecular dynamics simulations (optional for full workflow)
- **Plumed**: Advanced sampling methods (optional for full workflow)
- **Matplotlib/Seaborn**: Visualization
- **PyMOL/NGLView**: 3D molecular visualization
- **Pandas**: Data handling

## 3. Methodology Summary

### Core Concept
Reinforced Molecular Dynamics (rMD) combines molecular dynamics simulations with a physics-infused autoencoder to explore protein conformational space more efficiently. The key innovation is replacing the latent space of an autoencoder with a physical free energy map, thus providing physical context to the machine learning model.

### Workflow
1. **Collective Variable Definition**:
   - Define biologically relevant collective variables (CV) that capture the functional motion
   - For CRBN, these are the COM distances between key domains as shown in Fig. S1

2. **Free Energy Map Generation**:
   - Use advanced sampling MD simulations (meta-eABF) to generate free energy map in CV space
   - Process MD trajectories to extract conformations and their CV values

3. **Dual-Loss Autoencoder Training**:
   - Train an autoencoder on MD trajectory frames with two simultaneous loss functions:
     - Loss1 (latentLoss): Minimizes distance between latent space coordinates and CV coordinates
     - Loss2 (predLoss): Minimizes RMSD between input structures and reconstructed structures
   - This creates a one-to-one mapping between latent space and CV space

4. **Transition Path Exploration**:
   - Use the free energy map directly to identify low-energy paths
   - Sample points along paths (using B-spline interpolation)
   - Generate all-atom structures using the trained autoencoder
   - Post-process structures to remove local distortions

### Innovation
The key insight of rMD is the physical meaning injected into the latent space, allowing the direct use of free energy maps for structure generation and path exploration. This approach is entirely self-contained, does not rely on pre-trained models, and can be run on a single GPU desktop computer.

## Documentation Structure

### Introduction
Reinforced Molecular Dynamics (rMD) is a machine learning-based approach for modeling biologically relevant protein motions. It combines molecular dynamics trajectory data with free-energy map data to train a dual-loss function autoencoder network that can explore conformational space more efficiently than conventional MD simulations.

### Methodology
The core methodology involves:
1. Defining collective variables (CVs) that capture the biological motion of interest
2. Generating a free energy map over the CV space using enhanced sampling methods
3. Training an autoencoder with a dual-loss function that links the latent space to the CV space
4. Using the trained model to generate new conformations along transition pathways

### Dependencies
- Python 3.9+
- NumPy
- SciPy
- PyTorch
- MDAnalysis
- Matplotlib/Seaborn
- (Optional) OpenMM
- (Optional) Plumed

### Project Structure
```
reinforced_md/
├── collective_variables/     # CV definitions and calculations
├── data/                     # Data handling and processing
├── model/                    # Autoencoder implementation
├── sampling/                 # Path sampling and structure generation
├── visualization/            # Plotting and visualization tools
├── utils/                    # Utility functions
```

### Tests
Tests are implemented using pytest and cover:
1. CV calculations
2. Autoencoder functionality
3. Path finding algorithms
4. Structure generation and validation