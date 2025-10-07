# Contribution Guidelines

## Code Standards
- PEP8 compliance enforced via pre-commit hooks
- Type hints required for all functions
- Docstrings following Google style guide

## Workflow Process
1. Create feature branch from `dev`
2. Submit PR with:
   - Unit tests coverage >90%
   - Validation against reference MD data
   - Documentation updates
3. Require 2 approvals from:
   - Core developer
   - QA specialist

## Validation Protocol
```python
def validate_structure(predicted, reference):
    """
    Ensures predicted structures meet quality thresholds
    
    Args:
        predicted (MDAnalysis.Universe): Generated structure
        reference (MDAnalysis.Universe): MD reference
    
    Returns:
        bool: Pass/Fail status
    """
    # Implementation details omitted
```