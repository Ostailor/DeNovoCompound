# Hyperparameters and file paths
DATA_PATH    = None         # to be provided via CLI
MODEL_DIR    = "models"    # where to save/load models and vocab

# Model hyperparameters
EMB_DIM      = 128
HID_DIM      = 256
Z_DIM        = 64

# Training hyperparameters
BATCH_SIZE   = 64
LR_VAE       = 1e-3
LR_RL        = 1e-5

# Composite reward weights (tune as needed)
REWARD_WEIGHTS = {
    'qed': 1.0,
    'logp': 1.0,
    'sa': 1.0,
}