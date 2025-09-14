#!/usr/bin/env python3
"""
YOLOv8 Classification Training Script for DirtyCar Detection
Uses ultralytics YOLOv8 for binary classification (clean/dirty cars)
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data_yaml(data_root: str, output_path: str = "data.yaml"):
    """Create data.yaml file for YOLO training."""
    data_root = Path(data_root)
    
    # Check if train/val structure exists
    if (data_root / "train").exists():
        train_path = str(data_root / "train")
        val_path = str(data_root / "val") if (data_root / "val").exists() else str(data_root / "train")
    else:
        # Use root directory structure
        train_path = str(data_root)
        val_path = str(data_root)
    
    data_config = {
        'path': str(data_root.absolute()),
        'train': train_path,
        'val': val_path,
        'names': {
            0: 'clean',
            1: 'dirty'
        },
        'nc': 2  # number of classes
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logger.info(f"Created data.yaml: {output_path}")
    return output_path

def train_yolo_classifier(
    data_path: str,
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 224,
    batch_size: int = 32,
    device: str = "0",
    project: str = "runs/classify",
    name: str = "dirty_car_exp",
    resume: bool = False,
    **kwargs
):
    """Train YOLOv8 classification model."""
    
    # Initialize model
    model_name = f"yolov8{model_size}-cls.pt"
    logger.info(f"Loading model: {model_name}")
    
    model = YOLO(model_name)
    
    # Training arguments
    train_args = {
        'data': data_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        'resume': resume,
        'verbose': True,
        'save': True,
        'plots': True,
        'val': True,
        **kwargs
    }
    
    logger.info(f"Starting training with args: {train_args}")
    
    # Train the model
    results = model.train(**train_args)
    
    # Save final model
    model_save_path = Path(project) / name / "weights" / "best.pt"
    logger.info(f"Best model saved to: {model_save_path}")
    
    return results, model_save_path

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Classification for DirtyCar Detection')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--data_yaml', type=str, default=None,
                        help='Path to data.yaml file (will be created if not provided)')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (0, 1, cpu)')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='runs/classify',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='dirty_car_exp',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    
    # Additional YOLO arguments
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Validate data directory
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        sys.exit(1)
    
    # Create data.yaml if not provided
    if args.data_yaml is None:
        args.data_yaml = setup_data_yaml(args.data_root)
    
    # Set CUDA device
    if args.device != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Model size: yolov8{args.model_size}")
    
    # Additional training arguments
    additional_args = {
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
    }
    
    try:
        # Train model
        results, model_path = train_yolo_classifier(
            data_path=args.data_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            **additional_args
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        logger.info(f"Best model: {model_path}")
        
        # Create artifacts directory and copy best model
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        if model_path.exists():
            import shutil
            shutil.copy2(model_path, artifacts_dir / "best.pt")
            logger.info(f"Model copied to artifacts/best.pt")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
