"""
config.py
Configuration file for the Multi-Crop Disease Detection project.
"""

from pathlib import Path
import os

import torch


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data_split"

IMAGE_SIZE = 224
CHANNELS = 3

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
NUM_WORKERS = 0
PIN_MEMORY = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MODEL_DIR = BASE_DIR / "model"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
VAE_HEALTHY_THRESHOLD = float(os.environ.get("VAE_HEALTHY_THRESHOLD", "0.08"))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "vae_vit_baseline.pth"
