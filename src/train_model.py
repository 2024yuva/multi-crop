"""
train_model.py
Train hybrid VAE + classifier model.
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from src.dataset import get_dataloaders
from src.hybrid_model import HybridModel


MODEL_PATH = Path("model/best_hybrid_model.pth")


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def validate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _, _, _ = model(images)

            loss = classification_loss(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train():
    _, _, _, train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = HybridModel(num_classes=45).to(DEVICE)

    # Load pretrained VAE weights
    vae_weights = torch.load(
        "model/vae_encoder.pth",
        map_location=DEVICE
    )

    model.vae.load_state_dict(vae_weights, strict=False)

    # Freeze VAE
    for param in model.vae.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits, _, _, _ = model(images)

            loss = classification_loss(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"Batch {batch_idx} | Loss={loss.item():.4f}"
                )

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        val_loss, val_acc = validate(model, val_loader)

        print(
            f"\nEpoch {epoch+1}/{NUM_EPOCHS}"
            f"\nTrain Loss : {avg_train_loss:.4f}"
            f"\nTrain Acc  : {train_acc:.2f}%"
            f"\nVal Loss   : {val_loss:.4f}"
            f"\nVal Acc    : {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model.")

    print("Training complete.")


if __name__ == "__main__":
    train()