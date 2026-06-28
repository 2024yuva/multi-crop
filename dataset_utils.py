import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset_Split"
BATCH_SIZE = 16


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_test_transform


def get_datasets(data_dir=DATA_DIR):
    train_transform, val_test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=val_test_transform
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR):
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader