# src/vit.py
import torch
import torch.nn as nn
from torchvision import models


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=45):
        super().__init__()

        backbone = models.mobilenet_v2(weights="DEFAULT")

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x