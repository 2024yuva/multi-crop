import random

import numpy as np
import torch

from config import BATCH_SIZE, DATASET_DIR as DATA_DIR
from dataset import get_dataloaders, get_datasets


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
