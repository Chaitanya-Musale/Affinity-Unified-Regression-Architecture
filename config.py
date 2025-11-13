"""
AURA Framework - Configuration Module
Global configuration and hyperparameters for the AURA model.
"""

import torch

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 8
ACCUMULATION_STEPS = 8  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS = 64
MAX_EPOCHS_STAGE_B = 50
MAX_EPOCHS_STAGE_C = 50
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-4
GRADIENT_CLIP_NORM = 1.0
WARMUP_STEPS = 500
L2_REG_WEIGHT = 1e-5

# =============================================================================
# MODEL ARCHITECTURE PARAMETERS
# =============================================================================
GNN_2D_NODE_DIM = 6
GNN_2D_EDGE_DIM = 3
HIDDEN_DIM = 128
PLM_HIDDEN_DIM = 320
OUTPUT_DIM = 1
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# =============================================================================
# DATA PROCESSING PARAMETERS
# =============================================================================
N_CONFORMERS = 5
ECFP_N_BITS = 2048
MAX_PROTEIN_LENGTH = 1024

# Tokenizer pad ID (will be updated after tokenizer initialization)
PLM_PAD_ID = 1

# =============================================================================
# NORMALIZATION CONFIGURATION
# =============================================================================
NORMALIZATION_METHOD = 'zscore'  # Options: 'zscore', 'minmax'

# =============================================================================
# PATHS CONFIGURATION (Google Colab defaults)
# =============================================================================
BASE_DIR = '/content/drive/MyDrive/research/AURA'
SPLITS_DIR = f'{BASE_DIR}/Data_splits'

# Data split paths
TRAIN_CSV = f'{SPLITS_DIR}/train_split.csv'
VAL_CSV = f'{SPLITS_DIR}/validation_split.csv'
TEST_GENERAL_CSV = f'{SPLITS_DIR}/test_split.csv'
TEST_REFINED_CSV = f'{SPLITS_DIR}/test_refined_2020.csv'
TEST_CASF_CSV = f'{SPLITS_DIR}/test_casf_2016.csv'

# Structure paths (multiple paths for different datasets)
STRUCTURE_PATHS = [
    f'{BASE_DIR}/Data/PDBbind_v2020_other_PL/v2020-other-PL',
    f'{BASE_DIR}/Data/PDBbind_v2020_refined/refined-set',
    f'{BASE_DIR}/Data/CASF-2016/CASF-2016/coreset'
]

# Output directory
OUTPUT_DIR = f'{BASE_DIR}/Normalized_Model/v1'

# Cache file names
CONFORMER_CACHE = 'all_conformers_multipath_v2.pkl'
ECFP_CACHE = 'all_ecfp_precomputed.pkl'
TOKEN_CACHE = 'protein_tokens_multipath_v2.pkl'
NORMALIZER_CACHE = 'affinity_normalizer.pkl'

# =============================================================================
# DATALOADER CONFIGURATION
# =============================================================================
NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# =============================================================================
# XGBOOST CONFIGURATION
# =============================================================================
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_RANDOM_STATE = 42
XGB_EARLY_STOPPING_ROUNDS = 20

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================
VISUALIZATION_FIGURE_SIZE = (16, 10)
VISUALIZATION_DPI = 100

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
PRINT_TRAINING_INFO = True
SAVE_TRAINING_HISTORY = True

def update_pad_id(pad_id):
    """Update the PLM pad token ID after tokenizer initialization."""
    global PLM_PAD_ID
    PLM_PAD_ID = pad_id

def print_config():
    """Print current configuration."""
    print("\n" + "="*60)
    print("AURA FRAMEWORK CONFIGURATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Mixed Precision Training: {USE_AMP}")
    print(f"Batch Size: {BATCH_SIZE} (Effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs (Stage B/C): {MAX_EPOCHS_STAGE_B}/{MAX_EPOCHS_STAGE_C}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Normalization Method: {NORMALIZATION_METHOD}")
    print(f"Number of Conformers: {N_CONFORMERS}")
    print("="*60 + "\n")
