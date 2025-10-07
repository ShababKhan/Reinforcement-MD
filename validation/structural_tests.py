import MDAnalysis as mda

def validate_rmsd(predicted, reference, threshold=2.0):
    """Validate predicted structures against MD references"""
    pred = mda.Universe(predicted)
    ref = mda.Universe(reference)
    return np.allclose(pred.trajectory.rmsd(ref), 0, atol=threshold)

def check_cv_mapping(ls_coords, cv_coords):
    """Verify LS-CV space correlation"""
    return np.corrcoef(ls_coords.T, cv_coords.T)[0,1] > 0.9