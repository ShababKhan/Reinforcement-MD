# config.py
import torch

# --- Blueprint Constants ---
INPUT_DIM = 9696  # Length-9696 vector (M5)
LS_DIM = 3        # Latent Space Dimension (M3)
N_ROUNDS = 10000  # Training Epochs (M10)
BATCH_SIZE = 64   # Batch Size (M10)

# Loss Weights (To be determined but set defaults for the framework)
# Exact ratio is application-specific, but needs definition for dual-loss (SP2.2)
WEIGHT_L1 = 1.0
WEIGHT_L2 = 1.0

# Optimizer (M9: "Adams" interpreted as Adam)
OPTIMIZER = torch.optim.Adam

# Activation Function (M2)
ACTIVATION_FN = torch.nn.functional.silu # Python equivalent for Swish implementation
