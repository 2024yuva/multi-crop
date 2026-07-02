"""
vae.py
Variational autoencoder for multi-crop disease images.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CHANNELS, DEVICE, IMAGE_SIZE  # noqa: E402


LATENT_DIM = 256
HIDDEN_DIM = 128
ENCODER_FEATURES = 64
DECODER_START_CHANNELS = 32
DECODER_SPATIAL_SIZE = IMAGE_SIZE // 4


class Encoder(nn.Module):
    def __init__(self, in_channels=CHANNELS, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, ENCODER_FEATURES, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc_hidden = nn.Linear(
            ENCODER_FEATURES * DECODER_SPATIAL_SIZE * DECODER_SPATIAL_SIZE,
            hidden_dim,
        )
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc_hidden(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, out_channels=CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(
            latent_dim,
            DECODER_START_CHANNELS * DECODER_SPATIAL_SIZE * DECODER_SPATIAL_SIZE,
        )
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(
                DECODER_START_CHANNELS,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                16,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, DECODER_START_CHANNELS, DECODER_SPATIAL_SIZE, DECODER_SPATIAL_SIZE)
        return self.deconvs(x)


class VAE(nn.Module):
    def __init__(self, in_channels=CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(out_channels=in_channels, latent_dim=latent_dim)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decode(z)
        return reconstruction, z_mean, z_log_var


def vae_loss(reconstruction, target, z_mean, z_log_var):
    reconstruction_loss = F.binary_cross_entropy(reconstruction, target, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss


def smoke_test():
    model = VAE().to(DEVICE)
    inputs = torch.randn(2, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    inputs = torch.sigmoid(inputs)

    reconstruction, z_mean, z_log_var = model(inputs)
    total_loss, reconstruction_loss, kl_loss = vae_loss(
        reconstruction,
        inputs,
        z_mean,
        z_log_var,
    )

    print(f"Device: {DEVICE}")
    print(f"Input shape: {tuple(inputs.shape)}")
    print(f"Reconstruction shape: {tuple(reconstruction.shape)}")
    print(f"z_mean shape: {tuple(z_mean.shape)}")
    print(f"z_log_var shape: {tuple(z_log_var.shape)}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {reconstruction_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")


if __name__ == "__main__":
    smoke_test()
