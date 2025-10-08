
### Core Utilities (`src/md_utils.py`)
The foundational utility module was implemented to provide common molecular analysis tools.

1.  **Configuration Parsing (`parse_config`)**: Implements robust reading of project configuration settings from a JSON file, standardizing project parameters.
2.  **Root Mean Square Deviation (RMSD)**: The `calculate_rmsd` function implements the computationally demanding, rotationally-invariant RMSD calculation using the **Kabsch algorithm** (SVD-based). This ensures the structural similarity metric is optimized for minimal deviation, a critical requirement for accurate molecular dynamics trajectory analysis. (Kabsch, W. (1976). Acta Cryst. A32, 922–923.)
