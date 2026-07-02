"""
train_vae.py
Train the variational autoencoder and save reconstruction previews.
"""

from pathlib import Path
import argparse
import random
import sys

import numpy as np
import torch
from torchvision.utils import save_image

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import BATCH_SIZE, DEVICE, LEARNING_RATE, MODEL_DIR, NUM_EPOCHS, SEED  # noqa: E402
from src.dataset import get_dataloaders  # noqa: E402
from src.vae import VAE, vae_loss  # noqa: E402


VAE_EPOCHS = min(NUM_EPOCHS, 5)
PREVIEW_SAMPLES = 4
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = Path(MODEL_DIR)
FALLBACK_OUTPUT_ROOT = PROJECT_ROOT / "vae_outputs"
LOG_EVERY = 100


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_output_root():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    probe_path = MODEL_ROOT / ".write_test"

    try:
        probe_path.write_text("ok", encoding="ascii")
        probe_path.unlink()
        return MODEL_ROOT
    except OSError:
        FALLBACK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print(
            f"Model directory is not writable from Python; using fallback output directory: "
            f"{FALLBACK_OUTPUT_ROOT}",
            flush=True,
        )
        return FALLBACK_OUTPUT_ROOT


def save_reconstruction_grid(model, reference_batch, epoch, reconstruction_dir):
    model.eval()
    with torch.no_grad():
        originals = reference_batch[:PREVIEW_SAMPLES].to(DEVICE)
        reconstructions, _, _ = model(originals)

    comparison = torch.cat([originals.cpu(), reconstructions.cpu()], dim=0)
    output_path = reconstruction_dir / f"epoch_{epoch:02d}.png"
    save_image(
        comparison,
        output_path,
        nrow=PREVIEW_SAMPLES,
    )
    print(f"Saved reconstruction preview: {output_path}")


def train_one_epoch(model, loader, optimizer, max_batches=None):
    model.train()
    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    processed_batches = 0

    for batch_index, (images, _) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = images.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        reconstructions, z_mean, z_log_var = model(images)
        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstructions,
            images,
            z_mean,
            z_log_var,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reconstruction += reconstruction_loss.item()
        total_kl += kl_loss.item()
        processed_batches += 1

        if batch_index == 1 or batch_index % LOG_EVERY == 0:
            print(
                f"  batch {batch_index} | "
                f"loss={loss.item():.4f} | "
                f"recon={reconstruction_loss.item():.4f} | "
                f"kl={kl_loss.item():.4f}",
                flush=True,
            )

    num_batches = processed_batches
    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VAE on crop images.")
    parser.add_argument("--epochs", type=int, default=VAE_EPOCHS)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()
    output_root = get_output_root()
    model_path = output_root / "vae_encoder.pth"
    reconstruction_dir = output_root / "vae_reconstructions"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, _, train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        normalize=False,
    )

    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training epochs: {args.epochs}")
    if args.max_batches is not None:
        print(f"Max batches per epoch: {args.max_batches}")

    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    reference_batch, _ = next(iter(val_loader))

    for epoch in range(1, args.epochs + 1):
        train_loss, reconstruction_loss, kl_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            max_batches=args.max_batches,
        )
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Reconstruction Loss: {reconstruction_loss:.4f} | "
            f"KL Loss: {kl_loss:.4f}",
            flush=True,
        )
        save_reconstruction_grid(model, reference_batch, epoch, reconstruction_dir)

    torch.save(model.state_dict(), model_path)
    print(f"Saved trained VAE: {model_path}")


if __name__ == "__main__":
    main()
