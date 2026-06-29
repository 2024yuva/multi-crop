"""
config.py
Configuration file for the Multi-Crop Disease Detection project
(Base Paper Replication: VAE + Vision Transformer)
"""

import torch
import os

# =====================================================
# DATASET PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "data_split")

TRAIN_DIR = {
    "corn": os.path.join(DATASET_DIR, "corn", "train"),
    "groundnut": os.path.join(DATASET_DIR, "groundnut", "train"),
    "pearlmillet": os.path.join(DATASET_DIR, "pearlmillet", "train"),
    "rice": os.path.join(DATASET_DIR, "rice", "train"),
    "soybean": os.path.join(DATASET_DIR, "soybean", "train"),
    "wheat": os.path.join(DATASET_DIR, "wheat", "train")
}

VAL_DIR = {
    "corn": os.path.join(DATASET_DIR, "corn", "val"),
    "groundnut": os.path.join(DATASET_DIR, "groundnut", "val"),
    "pearlmillet": os.path.join(DATASET_DIR, "pearlmillet", "val"),
    "rice": os.path.join(DATASET_DIR, "rice", "val"),
    "soybean": os.path.join(DATASET_DIR, "soybean", "val"),
    "wheat": os.path.join(DATASET_DIR, "wheat", "val")
}

TEST_DIR = {
    "corn": os.path.join(DATASET_DIR, "corn", "test"),
    "groundnut": os.path.join(DATASET_DIR, "groundnut", "test"),
    "pearlmillet": os.path.join(DATASET_DIR, "pearlmillet", "test"),
    "rice": os.path.join(DATASET_DIR, "rice", "test"),
    "soybean": os.path.join(DATASET_DIR, "soybean", "test"),
    "wheat": os.path.join(DATASET_DIR, "wheat", "test")
}

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMAGE_SIZE = 224

CHANNELS = 3

# =====================================================
# TRAINING SETTINGS
# =====================================================

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

NUM_EPOCHS = 5

NUM_WORKERS = 0

PIN_MEMORY = False

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42

# =====================================================
# MODEL SAVE PATH
# =====================================================

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_NAME = "vae_vit_baseline.pth"
