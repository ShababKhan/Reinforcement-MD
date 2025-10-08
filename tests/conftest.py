"""Pytest configuration and shared fixtures for rMD testing."""
import pytest
import numpy as np


@pytest.fixture(scope="session")
def mock_trajectory_data():
    """Generate mock trajectory frames (10000 x 9696)."""
    N_FRAMES = 10000
    VECTOR_SIZE = 9696
    
    mock_data = np.random.rand(N_FRAMES, VECTOR_SIZE).astype(np.float32) * 10.0
    mock_data[0:5000, :] += 1.0
    mock_data[5001:10000, :] += 5.0
    
    return mock_data


@pytest.fixture(scope="session")
def mock_cv_targets(mock_trajectory_data):
    """Generate mock CV coordinates (10000 x 3)."""
    N_FRAMES = mock_trajectory_data.shape[0]
    return np.random.rand(N_FRAMES, 3).astype(np.float32) * 5.0
