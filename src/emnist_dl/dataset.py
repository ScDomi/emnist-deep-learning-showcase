"""Dataset preparation utilities for EMNIST Balanced.

The project uses the official torchvision EMNIST dataset and builds balanced
train/validation/test splits for a selected character subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import random
import string

from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetConfig:
    data_root: str = "data"
    allowed_chars: str = string.digits + string.ascii_uppercase
    train_samples_per_class: int = 5000
    val_samples_per_class: int = 500
    test_samples_per_class: int = 1000
    seed: int = 42


class RemappedDataset(Dataset):
    def __init__(self, base: Dataset, indices: list[int], labels: list[int], transform=None):
        self.base = base
        self.indices = indices
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, _ = self.base[self.indices[idx]]
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class BalancedDataset(Dataset):
    """Class-balanced view with optional replacement and augmentation."""

    def __init__(self, source: Dataset, class_to_indices: dict[int, list[int]], samples_per_class: int, transform, augment=None, seed: int = 42):
        self.source = source
        self.transform = transform
        self.augment = augment
        rng = random.Random(seed)
        self.samples: list[tuple[int, int, bool]] = []
        for cls, indices in sorted(class_to_indices.items()):
            if not indices:
                raise ValueError(f"Class {cls} has no samples")
            for i in range(samples_per_class):
                source_idx = indices[i % len(indices)]
                is_augmented = i >= len(indices)
                self.samples.append((source_idx, cls, is_augmented))
            rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        source_idx, label, is_augmented = self.samples[idx]
        image, _ = self.source[source_idx]
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
        if is_augmented and self.augment:
            image = self.augment(image)
        return self.transform(image), label


def emnist_label_to_char(label: int) -> str:
    # EMNIST Balanced labels: 0-9, A-Z, a-z with selected merged classes.
    chars = string.digits + string.ascii_uppercase + string.ascii_lowercase
    return chars[label]


def build_transforms(train: bool = False):
    base = [transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    augment = None
    if train:
        augment = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.90, 1.10), shear=8),
            transforms.RandomRotation(10),
        ])
    return transforms.Compose(base), augment


def _filter(base: Dataset, allowed_chars: Iterable[str], char_to_idx: dict[str, int]) -> tuple[list[int], list[int]]:
    allowed = set(allowed_chars)
    indices, labels = [], []
    for idx, (_, raw_label) in enumerate(base):
        char = emnist_label_to_char(int(raw_label))
        if char in allowed:
            indices.append(idx)
            labels.append(char_to_idx[char])
    return indices, labels


def _class_index(labels: list[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        out.setdefault(label, []).append(idx)
    return out


def get_datasets(config: DatasetConfig = DatasetConfig(), download: bool = True):
    allowed_chars = list(config.allowed_chars)
    idx_to_char = allowed_chars
    char_to_idx = {char: idx for idx, char in enumerate(idx_to_char)}

    train_raw = datasets.EMNIST(root=config.data_root, split="balanced", train=True, download=download)
    test_raw = datasets.EMNIST(root=config.data_root, split="balanced", train=False, download=download)

    train_indices, train_labels = _filter(train_raw, allowed_chars, char_to_idx)
    test_indices, test_labels = _filter(test_raw, allowed_chars, char_to_idx)

    eval_transform, _ = build_transforms(train=False)
    train_transform, augment = build_transforms(train=True)

    filtered_train = RemappedDataset(train_raw, train_indices, train_labels)
    filtered_test = RemappedDataset(test_raw, test_indices, test_labels)

    train_by_class = _class_index(train_labels)
    test_by_class = _class_index(test_labels)

    # Validation is held out from the filtered official training split.
    rng = random.Random(config.seed)
    val_source_indices: dict[int, list[int]] = {}
    train_source_indices: dict[int, list[int]] = {}
    for cls, indices in train_by_class.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        val_source_indices[cls] = shuffled[: config.val_samples_per_class]
        train_source_indices[cls] = shuffled[config.val_samples_per_class :]

    return {
        "train": BalancedDataset(filtered_train, train_source_indices, config.train_samples_per_class, train_transform, augment, config.seed),
        "val": BalancedDataset(filtered_train, val_source_indices, config.val_samples_per_class, eval_transform, None, config.seed),
        "test": BalancedDataset(filtered_test, test_by_class, config.test_samples_per_class, eval_transform, None, config.seed),
        "idx_to_char": idx_to_char,
        "char_to_idx": char_to_idx,
    }
