"""
dataset.py
Dataset and dataloader utilities for the Multi-Crop Disease Detection project.
"""

from pathlib import Path
from collections import Counter

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    DATASET_DIR,
    IMAGE_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
)


NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def _append_normalization(transform_steps, normalize):
    if normalize:
        transform_steps.append(
            transforms.Normalize(
                mean=NORMALIZE_MEAN,
                std=NORMALIZE_STD,
            )
        )


def _build_transforms(normalize=True):
    train_steps = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ]
    _append_normalization(train_steps, normalize)
    train_transform = transforms.Compose(train_steps)

    eval_steps = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
    _append_normalization(eval_steps, normalize)
    eval_transform = transforms.Compose(eval_steps)

    return train_transform, eval_transform


def _discover_classes(dataset_dir):
    dataset_path = Path(dataset_dir)
    class_names = set()

    for crop_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        train_dir = crop_dir / "train"
        if not train_dir.exists():
            continue
        for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
            class_names.add(class_dir.name)

    return sorted(class_names)


class MultiCropDiseaseDataset(Dataset):
    def __init__(self, split, dataset_dir=DATASET_DIR, transform=None, class_names=None):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self.classes = class_names or _discover_classes(self.dataset_dir)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for crop_dir in sorted(path for path in self.dataset_dir.iterdir() if path.is_dir()):
            split_dir = crop_dir / self.split
            if not split_dir.exists():
                continue

            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    continue

                for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                    if image_path.suffix.lower() in valid_suffixes:
                        samples.append((image_path, self.class_to_idx[class_name]))

        if not samples:
            raise ValueError(f"No images found for split '{self.split}' in '{self.dataset_dir}'.")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_datasets(dataset_dir=DATASET_DIR, normalize=True):
    train_transform, eval_transform = _build_transforms(normalize=normalize)
    class_names = _discover_classes(dataset_dir)

    train_dataset = MultiCropDiseaseDataset(
        split="train",
        dataset_dir=dataset_dir,
        transform=train_transform,
        class_names=class_names,
    )
    val_dataset = MultiCropDiseaseDataset(
        split="val",
        dataset_dir=dataset_dir,
        transform=eval_transform,
        class_names=class_names,
    )
    test_dataset = MultiCropDiseaseDataset(
        split="test",
        dataset_dir=dataset_dir,
        transform=eval_transform,
        class_names=class_names,
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size=BATCH_SIZE, dataset_dir=DATASET_DIR, normalize=True):
    train_dataset, val_dataset, test_dataset = get_datasets(
        dataset_dir=dataset_dir,
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def _count_labels(dataset):
    return Counter(label for _, label in dataset.samples)


def print_dataset_statistics(train_dataset, val_dataset, test_dataset):
    train_counts = _count_labels(train_dataset)
    val_counts = _count_labels(val_dataset)
    test_counts = _count_labels(test_dataset)

    print(f"Seed: {SEED}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("Per-class counts:")

    for class_index, class_name in enumerate(train_dataset.classes):
        print(
            f"  {class_name}: "
            f"train={train_counts[class_index]}, "
            f"val={val_counts[class_index]}, "
            f"test={test_counts[class_index]}"
        )


def display_sample_batch(train_dataset, train_loader):
    images, labels = next(iter(train_loader))
    label_names = [train_dataset.classes[label] for label in labels.tolist()]

    print(f"Batch shape: {tuple(images.shape)}")
    print(f"Labels: {labels.tolist()}")
    print(f"Label names: {label_names}")


def verify_data_pipeline():
    train_dataset, val_dataset, test_dataset, train_loader, _, _ = get_dataloaders()
    print_dataset_statistics(train_dataset, val_dataset, test_dataset)
    display_sample_batch(train_dataset, train_loader)


if __name__ == "__main__":
    verify_data_pipeline()
