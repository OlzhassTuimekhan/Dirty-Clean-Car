"""
Loss functions for DirtyCar binary classification.
Includes CrossEntropy with class weights and Focal Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            reduction: Specifies the reduction to apply to the output
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits tensor of shape (N, C) where N is batch size, C is number of classes
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """CrossEntropy loss with class weights for handling imbalanced datasets."""
    
    def __init__(self, class_weights: Optional[Tensor] = None, reduction: str = 'mean'):
        """
        Args:
            class_weights: Tensor of shape (C,) with weights for each class
            reduction: Specifies the reduction to apply to the output
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits tensor of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Weighted cross entropy loss
        """
        return F.cross_entropy(
            inputs, targets, 
            weight=self.class_weights, 
            reduction=self.reduction
        )


class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, class_weights: Optional[Tensor] = None):
        """
        Args:
            smoothing: Label smoothing factor (0.0 means no smoothing)
            class_weights: Optional class weights
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits tensor of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Label smoothed cross entropy loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Weight the true distribution
            weights = self.class_weights[targets].unsqueeze(1)
            true_dist = true_dist * weights
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def get_loss_function(loss_type: str, class_weights: Optional[Tensor] = None, **kwargs):
    """
    Factory function to get loss function based on configuration.
    
    Args:
        loss_type: Type of loss function ('ce', 'focal', 'label_smooth')
        class_weights: Optional class weights for balancing
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        Loss function instance
    """
    if loss_type == 'ce':
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'label_smooth':
        smoothing = kwargs.get('label_smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing, class_weights=class_weights)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    Reference: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On calibration of modern neural networks. ICML, 2017.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature parameter for scaling logits
        """
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: Tensor) -> Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model logits of shape (N, C)
        
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature
    
    def set_temperature(self, valid_loader, model, device):
        """
        Tune the temperature parameter using validation data.
        
        Args:
            valid_loader: Validation data loader
            model: Trained model
            device: Device to run on
        """
        model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        
        # Collect all logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        # Clamp temperature to reasonable range
        self.temperature.data = torch.clamp(self.temperature.data, min=0.1, max=10.0)
        
        return self.temperature.item()


if __name__ == "__main__":
    # Test loss functions
    batch_size = 8
    num_classes = 2
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test different loss functions
    print("Testing loss functions:")
    
    # Standard CrossEntropy
    ce_loss = nn.CrossEntropyLoss()
    print(f"CrossEntropy: {ce_loss(logits, targets).item():.4f}")
    
    # Weighted CrossEntropy
    class_weights = torch.tensor([0.3, 0.7])  # More weight on minority class
    weighted_ce = WeightedCrossEntropyLoss(class_weights)
    print(f"Weighted CrossEntropy: {weighted_ce(logits, targets).item():.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"Focal Loss: {focal_loss(logits, targets).item():.4f}")
    
    # Label Smoothing
    label_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
    print(f"Label Smoothing: {label_smooth(logits, targets).item():.4f}")
    
    # Temperature Scaling
    temp_scaling = TemperatureScaling(temperature=1.5)
    scaled_logits = temp_scaling(logits)
    print(f"Temperature scaled logits shape: {scaled_logits.shape}")
    print(f"Temperature: {temp_scaling.temperature.item():.4f}")
