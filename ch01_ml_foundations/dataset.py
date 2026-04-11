import os
import shutil
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


def verify_dataset_structure(data_dir: Path) -> bool:
    """
    Verify Kaggle chect X-ray dataset is correctly structured.
    Expected:   data_dir/train/NORMAL,  data_dir/train/PNEUMONIA,
                data_dir/val/NORMAL,    data_dir/val/PNEUMONIA,
                data_dir/test/NORMAL,   data_dir/test/PNEUMONIA
    """
    expected = [
        "train/NORMAL", "train/PNEUMONIA",
        "val/NORMAL",   "val/PNEUMONIA",
        "test/NORMAL",  "test/PNEUMONIA"
    ]

    missing = [p for p in expected if not (data_dir / p).exists()]
    if missing:
        print(f"Missing directories: {missing}")
        return False
    print("Dataset structure verified.")

    return True


def count_images(data_dir: Path) -> dict:
    """
    Count images per split and class.
    Returns dict with counts for reporting and imbalance calculation.
    """
    counts = {}

    for split in ["train", "val", "test"]:
        counts[split] = {}
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = data_dir / split / cls
            counts[split][cls] = len(list(cls_dir.glob("*.jpeg")))
    
    return counts

def report_dataset_stats(counts: dict) -> None:
    """
    Print dataset statistics and class imbalance ratio.
    """
    print("\nDataset Statistics:")
    print("-" * 40)

    for split in ["train", "val", "test"]:
        total = sum(counts[split].values())
        normal = counts[split]["NORMAL"]
        pneumonia = counts[split]["PNEUMONIA"]
        ratio = pneumonia / normal if normal > 0 else 0
        print(f"{split:>6}: {total:>5} images | "
              f"NORMAL: {normal:>4} | "
              f"PNEUMONIA: {pneumonia:>4} | "
              f"ratio: {ratio:.2f}")
    print("-" * 40)

def ChestXRayDataset(Dataset):
    """
    Pytorch Dataset for chest X-ray binary classification.
    Handles NORMAL (0) and PNEUMONIA (1) classes.
    """
    def __init__(
        self,
        data_dir: Path,
        split: str,
        transform=None
    ):
        self.data_dir = data_dir / split
        self.transform = transform
        self.samples = []
        self.labels = []

        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            cls_dir = self.data_dir / cls
            for img_path in sorted(cls_dir.glob("*.jpeg")):
                self.samples.append(img_path)
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def make_weighted_sampler(dataset: ChestXRayDataset) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler to correct class imbalance.
        Ensures each training batch sees roughly equal NORMAL/PNEUMONIA.
        """
        labels = np.array(dataset.labels)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler
    
    def get_dataloaders(
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build, train, val, and test DataLoaders.
        Train loader uses WeightedRandomSampler for class balance.
        Val and test loaders use sequential sampling, no shuffling.
        """
        from transformers import get_train_transforms, get_eval_transforms

        train_dataset = ChestXRayDataset(data_dir, "train", get_train_transforms())
        val_dataset = ChestXRayDataset(data_dir, "val", get_eval_transforms())
        test_dataset = ChestXRayDataset(data_dir, "test", get_eval_transforms())

        train_sampler = make_weighted_sampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader