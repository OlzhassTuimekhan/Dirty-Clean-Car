"""
Data module for DirtyCar binary classification with class imbalance handling.
Supports WeightedRandomSampler and oversampling strategies.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2


class AlbumentationsDataset(Dataset):
    """Dataset wrapper for Albumentations transforms."""
    
    def __init__(self, dataset: ImageFolder, transform: Optional[A.Compose] = None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


class OversamplingSampler:
    """Custom sampler for oversampling minority class."""
    
    def __init__(self, targets: List[int], oversample_ratio: float = 1.0):
        """
        Args:
            targets: List of class labels
            oversample_ratio: Target ratio of minority:majority class
        """
        self.targets = targets
        self.oversample_ratio = oversample_ratio
        
        # Count samples per class
        class_counts = Counter(targets)
        self.num_classes = len(class_counts)
        
        # Assume class 0 is majority (clean), class 1 is minority (dirty)
        self.majority_count = class_counts[0]  # clean
        self.minority_count = class_counts[1]  # dirty
        
        # Calculate target counts
        target_minority_count = int(self.majority_count * oversample_ratio)
        
        # Create indices for each class
        self.majority_indices = [i for i, label in enumerate(targets) if label == 0]
        self.minority_indices = [i for i, label in enumerate(targets) if label == 1]
        
        # Oversample minority class
        oversample_factor = target_minority_count // self.minority_count
        remainder = target_minority_count % self.minority_count
        
        self.oversampled_minority = (
            self.minority_indices * oversample_factor + 
            self.minority_indices[:remainder]
        )
        
        # Combine all indices
        self.all_indices = self.majority_indices + self.oversampled_minority
        
        print(f"Original distribution: clean={self.majority_count}, dirty={self.minority_count}")
        print(f"After oversampling: clean={len(self.majority_indices)}, dirty={len(self.oversampled_minority)}")
        print(f"Total samples: {len(self.all_indices)}")
    
    def __iter__(self):
        # Shuffle indices
        indices = self.all_indices.copy()
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.all_indices)


class DirtyCarDataModule:
    """Data module for DirtyCar classification with class balancing."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_root = Path(config['data_root'])
        self.img_size = config['img_size']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.pin_memory = config.get('pin_memory', True)
        self.persistent_workers = config.get('persistent_workers', True)
        
        # Class balancing configuration
        self.sampler_type = config.get('sampler', 'weighted')  # 'weighted', 'oversample', 'none'
        self.oversample_ratio = config.get('oversample_ratio', 1.0)
        
        # Augmentation config
        self.aug_config = config.get('augmentation', {})
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Class information
        self.class_names = config.get('class_names', ['clean', 'dirty'])
        self.num_classes = len(self.class_names)
        self.class_weights = None
        
    def setup(self):
        """Setup datasets and calculate class statistics."""
        # Create transforms
        train_transform = self._get_train_transforms()
        val_transform = self._get_val_transforms()
        
        # Load datasets
        train_folder = ImageFolder(str(self.data_root / 'train'))
        val_folder = ImageFolder(str(self.data_root / 'val'))
        
        self.train_dataset = AlbumentationsDataset(train_folder, train_transform)
        self.val_dataset = AlbumentationsDataset(val_folder, val_transform)
        
        # Load test dataset if exists
        test_path = self.data_root / 'test'
        if test_path.exists():
            test_folder = ImageFolder(str(test_path))
            self.test_dataset = AlbumentationsDataset(test_folder, val_transform)
        
        # Calculate class statistics
        self._calculate_class_stats()
        
        print(f"Dataset loaded:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        if self.test_dataset:
            print(f"  Test: {len(self.test_dataset)} samples")
        print(f"  Classes: {self.class_names}")
        print(f"  Class distribution (train): {dict(Counter(self.train_dataset.dataset.targets))}")
    
    def _get_train_transforms(self) -> A.Compose:
        """Create training transforms with augmentation."""
        transforms_list = [
            A.Resize(self.img_size, self.img_size),
        ]
        
        # Add augmentations
        if self.aug_config.get('horizontal_flip', 0) > 0:
            transforms_list.append(
                A.HorizontalFlip(p=self.aug_config['horizontal_flip'])
            )
        
        if self.aug_config.get('rotation', 0) > 0:
            transforms_list.append(
                A.Rotate(
                    limit=self.aug_config['rotation'],
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5
                )
            )
        
        if 'color_jitter' in self.aug_config:
            cj = self.aug_config['color_jitter']
            transforms_list.append(
                A.ColorJitter(
                    brightness=cj.get('brightness', 0),
                    contrast=cj.get('contrast', 0),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0),
                    p=0.5
                )
            )
        
        if self.aug_config.get('gaussian_blur', 0) > 0:
            transforms_list.append(
                A.GaussianBlur(blur_limit=(3, 7), p=self.aug_config['gaussian_blur'])
            )
        
        # Add normalization and tensor conversion
        norm_config = self.aug_config.get('normalize', {})
        transforms_list.extend([
            A.Normalize(
                mean=norm_config.get('mean', [0.485, 0.456, 0.406]),
                std=norm_config.get('std', [0.229, 0.224, 0.225])
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms_list)
    
    def _get_val_transforms(self) -> A.Compose:
        """Create validation/test transforms (no augmentation)."""
        norm_config = self.aug_config.get('normalize', {})
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=norm_config.get('mean', [0.485, 0.456, 0.406]),
                std=norm_config.get('std', [0.229, 0.224, 0.225])
            ),
            ToTensorV2()
        ])
    
    def _calculate_class_stats(self):
        """Calculate class weights and statistics."""
        targets = self.train_dataset.dataset.targets
        class_counts = Counter(targets)
        
        # Calculate class weights (inverse frequency)
        total_samples = len(targets)
        self.class_weights = []
        
        for i in range(self.num_classes):
            weight = total_samples / (self.num_classes * class_counts[i])
            self.class_weights.append(weight)
        
        self.class_weights = torch.FloatTensor(self.class_weights)
        
        print(f"Class weights: {dict(zip(self.class_names, self.class_weights.tolist()))}")
    
    def _get_weighted_sampler(self) -> WeightedRandomSampler:
        """Create weighted random sampler for class balancing."""
        targets = self.train_dataset.dataset.targets
        
        # Calculate sample weights
        class_counts = Counter(targets)
        sample_weights = []
        
        for target in targets:
            weight = 1.0 / class_counts[target]
            sample_weights.append(weight)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def _get_oversampling_sampler(self) -> OversamplingSampler:
        """Create oversampling sampler."""
        targets = self.train_dataset.dataset.targets
        return OversamplingSampler(targets, self.oversample_ratio)
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with appropriate sampler."""
        sampler = None
        shuffle = True
        
        if self.sampler_type == 'weighted':
            sampler = self._get_weighted_sampler()
            shuffle = False
        elif self.sampler_type == 'oversample':
            sampler = self._get_oversampling_sampler()
            shuffle = False
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader if test dataset exists."""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss function."""
        return self.class_weights


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Test data module
    config = load_config("configs/train.yaml")
    
    data_module = DirtyCarDataModule(config)
    data_module.setup()
    
    # Test dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"\nDataLoader test:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        print(f"Labels in batch: {labels.tolist()}")
        if batch_idx >= 2:
            break
