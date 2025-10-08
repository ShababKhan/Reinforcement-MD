# Mock Data Proxy for MD Trajectory and CVs

import numpy as np

# Configuration based on paper description
NUM_FRAMES = 10000
COORD_DIM = 9696  # Heavy atoms in CRBN structure
CV_DIM = 3

# --- Mock Coordinate Data ---
# Simulating flattened Cartesian coordinates of 10,000 frames.
# The values here are random but preserve the required dimensionality (10000, 9696).
# In a real scenario, these are floats derived from PDB/trajectory files.
np.random.seed(42)
mock_coords = np.random.rand(NUM_FRAMES, COORD_DIM).astype(np.float32) * 10.0

# --- Mock CV Data ---
# Simulating the 3 Collective Variables corresponding to those 10,000 frames.
# These should ideally form a distribution similar to the FE map.
# We create a simple clustering to mimic open/closed regions.
cv1_open = np.random.normal(loc=-5.0, scale=1.5, size=NUM_FRAMES)
cv1_closed = np.random.normal(loc=5.0, scale=1.0, size=NUM_FRAMES)
cv1 = np.concatenate((cv1_open[:NUM_FRAMES//2], cv1_closed[NUM_FRAMES//2:]))
cv2 = np.random.normal(loc=0.0, scale=1.0, size=NUM_FRAMES)
cv3 = np.random.normal(loc=0.0, scale=1.0, size=NUM_FRAMES)

mock_cvs = np.stack((cv1, cv2, cv3), axis=1).astype(np.float32)

# Save mock data
np.save('mock_trajectory_coords.npy', mock_coords)
np.save('mock_cv_targets.npy', mock_cvs)

print(f"Mock coordinate data shape: {mock_coords.shape}")
print(f"Mock CV data shape: {mock_cvs.shape}")
print("Mock data generation complete. Filenames: mock_trajectory_coords.npy, mock_cv_targets.npy")