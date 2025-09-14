"""
Model definitions for DirtyCar binary classification.
Supports timm models with customizable architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


class DirtyCarClassifier(nn.Module):
    """Binary classifier for car cleanliness detection."""
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        **kwargs
    ):
        """
        Args:
            model_name: Name of the timm model to use
            num_classes: Number of output classes (2 for binary)
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for the classifier head
        """
        super(DirtyCarClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Create backbone model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            **kwargs
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary predictions based on threshold.
        
        Args:
            x: Input tensor
            threshold: Threshold for clean class probability
        
        Returns:
            Binary predictions (0=clean, 1=dirty)
        """
        probs = self.predict_proba(x)
        clean_probs = probs[:, 0]  # Assuming class 0 is clean
        return (clean_probs < threshold).long()


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple models for improved performance."""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        super(EnsembleClassifier, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.register_buffer('weights', torch.tensor(weights))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble forward pass with weighted averaging."""
        outputs = []
        
        for model in self.models:
            with torch.no_grad():
                output = F.softmax(model(x), dim=1)
                outputs.append(output)
        
        # Weighted average
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += self.weights[i] * output
        
        return torch.log(ensemble_output + 1e-8)  # Convert back to log-probs


def create_model(config: Dict) -> DirtyCarClassifier:
    """
    Factory function to create model from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Model instance
    """
    model_config = {
        'model_name': config.get('model', 'resnet50'),
        'num_classes': config.get('num_classes', 2),
        'pretrained': config.get('pretrained', True),
        'dropout': config.get('dropout', 0.2)
    }
    
    return DirtyCarClassifier(**model_config)


def load_model(checkpoint_path: str, config: Dict, device: str = 'cpu') -> DirtyCarClassifier:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def save_model_info(model: DirtyCarClassifier, save_path: str):
    """Save model architecture information."""
    info = {
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        'feature_dim': model.feature_dim,
        'dropout': model.dropout,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)


class ModelEMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: Model to track
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.ema_model = None
        self.updates = 0
        
        # Create EMA model
        self._create_ema_model()
    
    def _create_ema_model(self):
        """Create EMA model copy."""
        self.ema_model = type(self.model)(
            model_name=self.model.model_name,
            num_classes=self.model.num_classes,
            pretrained=False,
            dropout=self.model.dropout
        )
        
        # Copy initial weights
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.eval()
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    def update(self):
        """Update EMA weights."""
        self.updates += 1
        
        # Adjust decay based on number of updates
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.mul_(decay).add_(model_param, alpha=1 - decay)
    
    def get_model(self) -> nn.Module:
        """Get EMA model."""
        return self.ema_model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def freeze_backbone(model: DirtyCarClassifier, freeze: bool = True):
    """Freeze or unfreeze backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """Get model summary information."""
    total_params, trainable_params = count_parameters(model)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'input_size': input_size
    }
    
    return summary


if __name__ == "__main__":
    # Test model creation
    config = {
        'model': 'resnet50',
        'num_classes': 2,
        'pretrained': True,
        'dropout': 0.2
    }
    
    print("Testing model creation...")
    model = create_model(config)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test probabilities
    probs = model.predict_proba(dummy_input)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs.sum(dim=1)}")
    
    # Model summary
    summary = get_model_summary(model)
    print(f"\nModel Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test different models
    models_to_test = ['resnet50', 'efficientnet_b0', 'convnext_tiny']
    
    for model_name in models_to_test:
        try:
            test_config = config.copy()
            test_config['model'] = model_name
            test_model = create_model(test_config)
            total_params, _ = count_parameters(test_model)
            print(f"{model_name}: {total_params:,} parameters")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
