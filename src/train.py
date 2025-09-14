"""
Training script for DirtyCar binary classification with DDP, AMP, and advanced features.
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# Local imports
from data_module import DirtyCarDataModule, load_config
from model import create_model, ModelEMA, count_parameters
from losses import get_loss_function, TemperatureScaling


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, mode: str = 'max', min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class MetricsCalculator:
    """Calculate and track training metrics."""
    
    def __init__(self, num_classes: int = 2, class_names: list = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Per-class metrics
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision[i] if i < len(precision) else 0.0
            metrics[f'{class_name}_recall'] = recall[i] if i < len(recall) else 0.0
            metrics[f'{class_name}_f1'] = f1[i] if i < len(f1) else 0.0
        
        return metrics


class DirtyCarTrainer:
    """Main trainer class for DirtyCar classification."""
    
    def __init__(self, config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        
        # Setup paths
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './artifacts'))
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.tensorboard_dir = Path(config.get('tensorboard_dir', './runs'))
        
        # Create directories
        if self.is_main_process:
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.log_dir.mkdir(exist_ok=True)
            self.tensorboard_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        self.data_module = None
        self.writer = None
        self.ema = None
        self.temperature_scaling = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf') if config.get('early_stopping', {}).get('mode', 'max') == 'max' else float('inf')
        
        # Metrics
        self.metrics_calc = MetricsCalculator(
            num_classes=config.get('num_classes', 2),
            class_names=config.get('class_names', ['clean', 'dirty'])
        )
        
        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            mode=early_stop_config.get('mode', 'max'),
            min_delta=early_stop_config.get('min_delta', 0.0)
        )
        
        self.setup()
    
    def setup(self):
        """Setup all components."""
        # Setup distributed training
        if self.world_size > 1:
            self._setup_distributed()
        
        # Setup data
        self.data_module = DirtyCarDataModule(self.config)
        self.data_module.setup()
        
        # Setup model
        self.model = create_model(self.config)
        self.model.to(self.device)
        
        if self.is_main_process:
            total_params, trainable_params = count_parameters(self.model)
            print(f"Model: {self.config.get('model', 'resnet50')}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Setup DDP
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
        
        # Setup loss function
        class_weights = None
        if self.config.get('class_weights') == 'auto':
            class_weights = self.data_module.get_class_weights().to(self.device)
        elif isinstance(self.config.get('class_weights'), list):
            class_weights = torch.tensor(self.config['class_weights']).to(self.device)
        
        self.criterion = get_loss_function(
            self.config.get('loss', 'ce'),
            class_weights=class_weights,
            focal_alpha=self.config.get('focal_alpha', 0.25),
            focal_gamma=self.config.get('focal_gamma', 2.0)
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('lr', 3e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Setup scheduler
        self._setup_scheduler()
        
        # Setup AMP
        if self.config.get('amp', True):
            self.scaler = GradScaler()
        
        # Setup EMA
        if self.config.get('use_ema', False):
            self.ema = ModelEMA(self.model.module if hasattr(self.model, 'module') else self.model)
        
        # Setup temperature scaling
        if self.config.get('temperature_scaling', False):
            self.temperature_scaling = TemperatureScaling()
        
        # Setup tensorboard
        if self.is_main_process:
            self.writer = SummaryWriter(self.tensorboard_dir)
    
    def _setup_distributed(self):
        """Setup distributed training."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self.config.get('master_port', 29500))
        dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 40),
                eta_min=self.config.get('lr', 3e-4) * 0.01
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 10),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=5,
                factor=0.5,
                verbose=True
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        train_loader = self.data_module.train_dataloader()
        
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=not self.is_main_process)
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            # Collect predictions
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
            running_loss += loss.item()
            self.global_step += 1
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 50) == 0 and self.is_main_process:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(train_loader)
        metrics = self.metrics_calc.calculate_metrics(
            np.array(all_targets), 
            np.array(all_preds), 
            np.array(all_probs)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        val_loader = self.data_module.val_dataloader()
        
        running_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation', disable=not self.is_main_process):
                images, targets = images.to(self.device), targets.to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                running_loss += loss.item()
        
        # Calculate metrics
        avg_loss = running_loss / len(val_loader)
        metrics = self.metrics_calc.calculate_metrics(
            np.array(all_targets), 
            np.array(all_preds), 
            np.array(all_probs)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
        
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config,
            'best_metric': self.best_metric
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.get_model().state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.ckpt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.ckpt')
            
            # Save model for inference
            model_for_inference = self.ema.get_model() if self.ema else (
                self.model.module if hasattr(self.model, 'module') else self.model
            )
            torch.save(model_for_inference.state_dict(), self.checkpoint_dir / 'best.pt')
            
            # Save scripted model
            model_for_inference.eval()
            scripted_model = torch.jit.script(model_for_inference)
            scripted_model.save(str(self.checkpoint_dir / 'best_scripted.pt'))
    
    def train(self):
        """Main training loop."""
        if self.is_main_process:
            print(f"Starting training for {self.config.get('epochs', 40)} epochs...")
            print(f"Device: {self.device}")
            print(f"World size: {self.world_size}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.get('epochs', 40) + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config.get('monitor_metric', 'macro_f1')])
                else:
                    self.scheduler.step()
            
            # Log metrics
            if self.is_main_process:
                self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Check for best model
            monitor_metric = self.config.get('monitor_metric', 'macro_f1')
            current_metric = val_metrics[monitor_metric]
            
            is_best = False
            if self.config.get('early_stopping', {}).get('mode', 'max') == 'max':
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            else:
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping(current_metric):
                if self.is_main_process:
                    print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Fit temperature scaling if enabled
        if self.temperature_scaling is not None and self.is_main_process:
            self._fit_temperature_scaling()
        
        # Final cleanup
        if self.is_main_process:
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Best {monitor_metric}: {self.best_metric:.4f}")
            
            if self.writer:
                self.writer.close()
        
        if self.world_size > 1:
            dist.destroy_process_group()
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to tensorboard and console."""
        # Console logging
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}")
        
        # Per-class metrics
        for class_name in self.config.get('class_names', ['clean', 'dirty']):
            train_f1 = train_metrics.get(f'{class_name}_f1', 0.0)
            val_f1 = val_metrics.get(f'{class_name}_f1', 0.0)
            print(f"  {class_name.capitalize()} F1 - Train: {train_f1:.4f}, Val: {val_f1:.4f}")
        
        # TensorBoard logging
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric_name}', value, epoch)
        
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
    
    def _fit_temperature_scaling(self):
        """Fit temperature scaling on validation data."""
        print("Fitting temperature scaling...")
        
        # Load best model
        best_model = create_model(self.config)
        best_model.load_state_dict(torch.load(self.checkpoint_dir / 'best.pt'))
        best_model.to(self.device)
        
        # Fit temperature
        val_loader = self.data_module.val_dataloader()
        temperature = self.temperature_scaling.set_temperature(val_loader, best_model, self.device)
        
        # Save temperature
        temp_info = {
            'temperature': temperature,
            'calibration_bins': self.config.get('calibration_bins', 15)
        }
        
        with open(self.checkpoint_dir / 'temperature.json', 'w') as f:
            json.dump(temp_info, f, indent=2)
        
        print(f"Temperature scaling fitted: T = {temperature:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train DirtyCar classifier')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Config file path')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup distributed training
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
    else:
        world_size = 1
        rank = 0
    
    # Create trainer and start training
    trainer = DirtyCarTrainer(config, rank=rank, world_size=world_size)
    trainer.train()


if __name__ == "__main__":
    main()
