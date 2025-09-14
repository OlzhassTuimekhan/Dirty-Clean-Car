"""
Visualization utilities for DirtyCar classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Optional
import itertools


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         normalize: bool = False, title: str = 'Confusion Matrix',
                         cmap: str = 'Blues', figsize: tuple = (8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize the matrix
        title: Plot title
        cmap: Colormap
        figsize: Figure size
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float,
                  title: str = 'ROC Curve', figsize: tuple = (8, 6)):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: AUC score
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)


def plot_precision_recall_curve(recall: np.ndarray, precision: np.ndarray, 
                               ap_score: float, title: str = 'Precision-Recall Curve',
                               figsize: tuple = (8, 6)):
    """
    Plot Precision-Recall curve.
    
    Args:
        recall: Recall values
        precision: Precision values
        ap_score: Average precision score
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {ap_score:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)


def plot_training_history(history: dict, save_path: Optional[str] = None,
                         figsize: tuple = (15, 5)):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_accuracy'], label='Train Accuracy', color='blue')
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[2].plot(history['train_f1'], label='Train F1', color='blue')
    axes[2].plot(history['val_f1'], label='Val F1', color='red')
    axes[2].set_title('Training and Validation F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(class_counts: dict, title: str = 'Class Distribution',
                          figsize: tuple = (10, 6)):
    """
    Plot class distribution.
    
    Args:
        class_counts: Dictionary with class counts
        title: Plot title
        figsize: Figure size
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    bars = ax1.bar(classes, counts, color=['skyblue', 'lightcoral'])
    ax1.set_title(f'{title} - Counts')
    ax1.set_ylabel('Number of Samples')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90,
            colors=['skyblue', 'lightcoral'])
    ax2.set_title(f'{title} - Proportions')
    
    plt.tight_layout()


def plot_threshold_analysis(thresholds: np.ndarray, precisions: np.ndarray,
                          recalls: np.ndarray, f1_scores: np.ndarray,
                          optimal_threshold: float, target_precision: float = 0.95,
                          figsize: tuple = (12, 8)):
    """
    Plot threshold analysis for precision-recall trade-off.
    
    Args:
        thresholds: Array of threshold values
        precisions: Precision values for each threshold
        recalls: Recall values for each threshold
        f1_scores: F1 scores for each threshold
        optimal_threshold: Selected optimal threshold
        target_precision: Target precision value
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Precision vs Threshold
    axes[0, 0].plot(thresholds, precisions, 'b-', label='Precision')
    axes[0, 0].axhline(y=target_precision, color='r', linestyle='--', 
                      label=f'Target Precision ({target_precision})')
    axes[0, 0].axvline(x=optimal_threshold, color='g', linestyle='--', 
                      label=f'Optimal Threshold ({optimal_threshold:.3f})')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall vs Threshold
    axes[0, 1].plot(thresholds, recalls, 'r-', label='Recall')
    axes[0, 1].axvline(x=optimal_threshold, color='g', linestyle='--', 
                      label=f'Optimal Threshold ({optimal_threshold:.3f})')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score vs Threshold
    axes[1, 0].plot(thresholds, f1_scores, 'g-', label='F1 Score')
    axes[1, 0].axvline(x=optimal_threshold, color='g', linestyle='--', 
                      label=f'Optimal Threshold ({optimal_threshold:.3f})')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Trade-off
    axes[1, 1].plot(recalls, precisions, 'purple', label='Precision-Recall Curve')
    axes[1, 1].axhline(y=target_precision, color='r', linestyle='--', 
                      label=f'Target Precision ({target_precision})')
    
    # Mark optimal point
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    axes[1, 1].plot(recalls[optimal_idx], precisions[optimal_idx], 'go', 
                   markersize=10, label=f'Optimal Point')
    
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision-Recall Trade-off')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()


def plot_prediction_distribution(clean_probs: np.ndarray, dirty_probs: np.ndarray,
                                threshold: float, figsize: tuple = (12, 5)):
    """
    Plot distribution of prediction probabilities.
    
    Args:
        clean_probs: Probabilities for clean samples
        dirty_probs: Probabilities for dirty samples
        threshold: Decision threshold
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram of probabilities
    axes[0].hist(clean_probs, bins=50, alpha=0.7, label='Clean (True)', 
                color='blue', density=True)
    axes[0].hist(dirty_probs, bins=50, alpha=0.7, label='Dirty (True)', 
                color='red', density=True)
    axes[0].axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold:.3f})')
    axes[0].set_xlabel('Predicted Probability (Clean)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Predicted Probabilities')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data = [clean_probs, dirty_probs]
    labels = ['Clean (True)', 'Dirty (True)']
    box_plot = axes[1].boxplot(data, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    
    axes[1].axhline(y=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold:.3f})')
    axes[1].set_ylabel('Predicted Probability (Clean)')
    axes[1].set_title('Box Plot of Predicted Probabilities')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()


if __name__ == "__main__":
    # Test visualization functions
    import numpy as np
    
    # Test confusion matrix
    cm = np.array([[85, 15], [20, 80]])
    class_names = ['Clean', 'Dirty']
    
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(cm, class_names, normalize=True)
    plt.show()
    
    # Test ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Dummy data
    auc_score = 0.85
    
    plot_roc_curve(fpr, tpr, auc_score)
    plt.show()
    
    # Test class distribution
    class_counts = {'Clean': 69000, 'Dirty': 2300}
    plot_class_distribution(class_counts)
    plt.show()
