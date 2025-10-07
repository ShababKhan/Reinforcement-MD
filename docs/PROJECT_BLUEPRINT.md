# CRBN Activation rMD Project Blueprint

## Agile Project Plan

### Sprint 1: MD Simulation Infrastructure
1. **User Story**: As a computational chemist, I need meta-eABF simulations to generate FE maps
   - Tasks:
     - Implement collective variables (CV1-3) for CRBN domains (Fig S1)
     - Configure AMBER ff14sb/gaff force fields with TIP3P water
     - Validate hydrogen mass repartitioning (HMR) implementation
   - Acceptance: 1μs simulation @135ns/day on RTX 4080 (Methods 2.1)

### Sprint 2: Autoencoder Core Architecture
2. **User Story**: As an ML engineer, I need physics-informed autoencoder for latent space mapping
   - Tasks:
     - Implement encoder (9696→64→32→3) with Swish activation (Fig S2)
     - Build symmetric decoder (3→32→64→9696)
     - Dual loss integration (MSE + CV alignment) from Eq 5
   - Acceptance: Loss1 ≤1Å, Loss2 ≤1.6Å (Results 3.2)

### Sprint 3: Conformational Sampling
3. **User Story**: As a biophysicist, I need path generation along FE landscapes
   - Tasks:
     - B-spline path interpolation in CV space (Fig 4)
     - Structure relaxation protocol (Methods 2.3)
     - Rosetta integration for local geometry fixes
   - Acceptance: Generate 20-state transition path (Movie S1)

## Component Stack

### Core Modules
- **MD Driver**: OpenMM-Plumed (v8.1/2.9)
- **ML Core**: PyTorch 2.3 + CUDA 12.4
- **Analysis**: MDAnalysis 2.8 + MDTraj 1.9

### Dependencies
```python
REQUIREMENTS = {
    'simulation': ['amber-md==24.0', 'plumed==2.9'],
    'ml': ['pytorch==2.3.0', 'numpy==1.26'],
    'viz': ['matplotlib==3.8', 'nglview==3.1']
}
```

## Methodology Documentation

### FE Map Generation Protocol
```markdown
1. Run 2×1μs meta-eABF simulations (Methods 2.1)
2. Compute 3D FE landscape via CZAR (Methods 2.2)
3. Grid resolution: 0.1Å CV spacing (Fig 3)
```

### Validation Framework
| Metric          | Target       | Method               | Paper Reference |
|-----------------|--------------|----------------------|-----------------|
| Latent-CV RMSD  | ≤1.0Å       | Procrustes analysis  | Fig 3          |
| Path Energy     | ΔG≤5kcal/mol | PMF integration      | Fig 4          |
| t-SNE Baseline  | KL-div≤0.1  | Dimensionality comparison | Table 2     |
```