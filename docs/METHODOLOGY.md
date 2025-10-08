# Methodology Summary: Reinforced Molecular Dynamics (rMD)

## High-Level Implementation Phases

... (Existing Content) ...

## Methodology Detail

### Data Ingestion and Structure Initialization (User Story 0.2)

**Code Module:** `src.data_loader`

**Function:** `load_pdb_structure(file_path)`

**Scientific Principle:** Molecular Dynamics simulations require a well-defined initial configuration, typically derived from experimental data like X-ray crystallography or NMR spectroscopy, stored in the Protein Data Bank (PDB) format.

**Implementation Detail:** This function implements the foundational step of reading and parsing a PDB flat file. It leverages the robust `Bio.PDB` library to convert the linear PDB record format into a hierarchical data structure (`Bio.PDB.Structure.Structure`: model -> chain -> residue -> atom). This structured representation is essential for all subsequent physical calculations (e.g., calculating forces, determining bonds, and analyzing geometry). The implementation ensures file existence checks and error handling during parsing, critical for scientific reproducibility.

### Feature Engineering: Collective Variable Calculation (User Story 1.1)

**Code Module:** `src.feature_engineering`

**Function:** `calculate_dihedral_angles(structure)`

**Scientific Principle (PI-AE Input):** The Physics-Informed Autoencoder (PI-AE) is designed to learn the underlying manifold of the molecular system's configuration space. For standard biopolymers like proteins, the primary degrees of freedom are the rotation around backbone bonds, defined by the dihedral (torsion) angles Phi ($\phi$), Psi ($\psi$), and Omega ($\omega$). These angles serve as the explicit, physically-motivated Collective Variables (CVs) used as input features to the PI-AE model.

**Implementation Detail:** This function iterates through the protein sequence (residue-by-residue) and utilizes the `Bio.PDB.calc_dihedral` utility to compute the standard $\phi, \psi,$ and $\omega$ backbone angles (in radians). It includes safeguards to handle the chain terminals (where $\phi$ or $\psi$ are undefined) and ensures that the required four atoms are present for each calculation—a critical step for robust processing of PDB files which may have missing atoms.

### Feature Engineering: Sinusoidal Encoding (User Story 1.2)

**Code Module:** `src.feature_engineering`

**Function:** `encode_dihedral_features(angles)`

**Scientific Principle (PI-AE Input Feature Continuity):** The PI-AE model requires input features to be continuous to ensure stable gradient descent during training. Dihedral angles ($\theta$) are periodic (e.g., $180^\circ$ is numerically far from $-180^\circ$, but physically identical). Standard machine learning algorithms struggle with this discontinuity. The widely accepted solution in MD machine learning is to transform the periodic angle $\theta$ into a two-dimensional Cartesian vector $(\cos\theta, \sin\theta)$. This transformation preserves the angular information while eliminating the discontinuity, resulting in a continuous, 6-dimensional feature vector per residue (2 dimensions each for $\phi, \psi, \omega$).

**Implementation Detail:** The function takes the calculated dihedral angles and applies the $\cos\theta, \sin\theta$ transformation. It carefully handles terminal residues or residues with missing atoms (where the angle value is `None`) by skipping them, ensuring only complete, valid feature vectors enter the PI-AE model.

### PI-AE Model Architecture and Initialization (User Story 1.3)

**Code Module:** `src.models`

**Class:** `PhysicsInformedAutoencoder`

**Scientific Principle (Dimensionality Reduction):** The core of the rMD method relies on finding a parsimonious set of generalized collective variables (gCVs) that accurately describe the slow dynamics of the system. The PI-AE achieves this by minimizing the difference between the input features (sinusoidal CVs) and the reconstructed output, forcing the intermediate bottleneck (latent space) to capture the most essential variances. The latent space dimensions (e.g., $N=2$) become the gCVs.

**Implementation Detail:** This class implements a standard feed-forward Autoencoder structure using PyTorch's `nn.Module`. The architecture is intentionally symmetric, with a defined `input_dim` of 6 (from US 1.2) and a low `latent_dim` (default 2). All hidden layers use the ReLU activation function to introduce non-linearity, and the network is designed to isolate the encoder and decoder components to allow for separate use when mapping configurations to gCVs (using the encoder) or projecting gCVs back to configuration space (using the decoder).

### Simulation Environment: System State and Integrator (User Story 2.1)

**Code Module:** `src.md_engine`

**Classes:** `SystemState`, `AbstractIntegrator`

**Scientific Principle (MD Foundations):** Molecular Dynamics simulation involves numerically solving Newton's equations of motion ($F=ma$) for a system of particles. The fundamental requirements are a well-defined state vector (positions $\mathbf{r}$, velocities $\mathbf{v}$, and forces $\mathbf{f}$) and a time-stepping algorithm (Integrator). MD methods, including meta-eABF, operate by iteratively updating the `SystemState` based on forces calculated from the system's potential energy function and any applied biases.

**Implementation Detail:** `SystemState` encapsulates the positions, velocities, forces, and masses as NumPy arrays, facilitating high-speed, vectorized calculations. `AbstractIntegrator` defines a clear interface (`step`) using the mandatory numerical time step ($\Delta t$). This abstraction ensures that specific integrator algorithms (like Velocity Verlet, to be implemented later) can be plugged in without changing the core MD flow designed for meta-eABF.

### Simulation Core: Velocity Verlet Integrator (User Story 2.2)

**Code Module:** `src.md_engine`

**Class:** `VelocityVerlet`

**Scientific Principle (Symplectic Integration):** For Molecular Dynamics simulations, numerical stability and thermodynamic consistency are achieved through Symplectic integrators like Velocity Verlet. This method ensures that the trajectory closely follows the true dynamics of the system Hamiltonian. The implementation splits the momentum update into two half-steps, sandwiching a full position update and the expensive force calculation step.

**Implementation Detail:** The `VelocityVerlet.step` function implements the standard three-part algorithm: half-step velocity update, full-step position update, and final half-step velocity update based on the new force $(\mathbf{f}')$. Crucially, it manages the division by mass by utilizing NumPy's broadcasting capabilities (`1.0 / state.masses[:, None]`) for vectorized and efficient acceleration calculation $(\mathbf{a} = \mathbf{f} / \mathbf{m})$. The implementation requires the new force ($\mathbf{f}_{new}$) to be supplied externally, mirroring the separation of integration and force evaluation inherent in the MD loop.

### Simulation Core: Force Calculation Interface (User Story 2.3)

**Code Module:** `src.force_provider`

**Classes:** `AbstractForceProvider`, `MockForceProvider`

**Scientific Principle (ABF Mechanism):** The core mechanism of the Adaptive Biasing Force (ABF) method is the addition of a biasing force ($\mathbf{f}_{\text{bias}}$) to the physical MD force ($\mathbf{f}_{\text{md}}$). This biasing force is defined as the negative gradient of the free energy surface along the chosen generalized collective variables (gCVs). To decouple the integrator (Verlet) from the force laws, a dedicated `ForceProvider` interface is necessary.

**Implementation Detail:** `AbstractForceProvider` defines the mandatory `calculate_force` and `get_potential_energy` methods, ensuring all force computation logic is centralized. `MockForceProvider` is a necessary placeholder that allows the MD simulation loop (to be implemented next) to be structurally tested without requiring the complexity of a full molecular potential or the meta-eABF bias implementation yet. It uses NumPy to return the total force vector $N \times 3$ for vectorization compatibility.

### Simulation Core: Molecular Dynamics Simulator (User Story 2.4)

**Code Module:** `src.md_engine`

**Class:** `MDSimulator`

**Scientific Principle (Core MD Cycle):** The `MDSimulator` encapsulates the standard MD time evolution cycle. The mandatory sequence is established: load forces (from ForceProvider), advance state (Integrator), record data. This structure ensures that potential energy, kinetic energy, and all biasing forces are calculated consistently within the control volume defined by the simulation loop.

**Implementation Detail:** The `MDSimulator.run_steps` method orchestrates the cycle. It initializes the system by calculating $F(t)$ and then iteratively advances time using the `VelocityVerlet.step` method. Data recording is implemented to capture the trajectory (positions $\mathbf{r}$) and the system's potential energy ($U$), which are critical observables for subsequent analysis and for training the PI-AE model. The current implementation uses a structural mock where $F(t+\Delta t)$ is calculated from $F(t)$, which is acceptable for validating the *loop flow* but acknowledges that a true MD engine would pass $\mathbf{r}(t+\Delta t)$ to the ForceProvider before the final velocity update.

### Meta-eABF Components: Generalized Collective Variable (gCV) Mapper (User Story 3.1)

**Code Module:** `src.collective_variables`

**Class:** `gCVMapper`

**Scientific Principle (gCV Identification):** The Adaptive Biasing Force (ABF) method requires the ability to continuously monitor the position of the system along the gCVs ($\xi$). The PI-AE component defines the functional form of this map ($E(\mathbf{r}) \rightarrow \xi$). The mapping is non-trivial: it involves extracting fundamental molecular coordinates (dihedrals, $\theta$), encoding them sinusoidally, and then passing them through the pretrained Autoencoder's encoder half.

**Implementation Detail:** `gCVMapper` orchestrates this multi-step mapping. It holds the reference `PhysicsInformedAutoencoder` and an abstract coordinate linker to manage the connection between the instantaneous MD positions (`SystemState.r`) and the molecular structure required by the feature engineering functions. The `r_to_xi` method transforms the coordinates using US 1.1 and US 1.2 logic, converts the data to a PyTorch tensor, applies the gCVs encoder (US 1.3), and returns the final low-dimensional $\xi$ values as a NumPy array.

### Meta-eABF Components: Adaptive Biasing Force Histogram (User Story 3.2)

**Code Module:** `src.abf_engine`

**Class:** `ABFHistogram`

**Scientific Principle (Free Energy Estimation):** The Adaptive Biasing Force (ABF) method estimates the free energy surface along the generalized collective variables ($\xi$) by accumulating the true mean force, $\langle \mathbf{F}_{\xi} \rangle$, in histogram bins. The biasing force applied to the system is then given by $\mathbf{F}_{bias} = -\langle \mathbf{F}_{\xi} \rangle$, which cancels the mean force and "flattens" the free energy landscape, allowing the system to explore high-energy regions.

**Implementation Detail:** The `ABFHistogram` class uses high-efficiency NumPy arrays to store the sample counts (`self.counts`) and the total accumulated force vector (`self.sum_forces`) for each bin. The `_get_bin_indices` method performs the critical mapping from continuous $\xi$ values to discrete integer bin indices. The `get_instantaneous_bias` method calculates the negative mean force on demand, implementing the core statistical component of the ABF method.
