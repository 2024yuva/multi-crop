# src/hybrid_model.py
import torch
import torch.nn as nn

from src.vae import VAE
from src.vit import ViTClassifier


class HybridModel(nn.Module):
    def __init__(self, num_classes=45):
        super().__init__()

        self.vae = VAE()
        self.classifier = ViTClassifier(num_classes)

    def forward(self, x):
        reconstructed, z_mean, z_log_var = self.vae(x)

        logits = self.classifier(reconstructed)

        return logits, reconstructed, z_mean, z_log_var