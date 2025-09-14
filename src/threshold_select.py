"""
Threshold selection for DirtyCar binary classification.
Optimizes threshold to achieve target precision for clean class.
"""

import os
import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from tqdm import tqdm

# Local imports
from data_module import DirtyCarDataModule, load_config
from model import load_model
from utils_vis import plot_threshold_analysis, plot_prediction_distribution


class ThresholdSelector:
    """Threshold selection for high precision clean class detection."""
    
    def __init__(self, config: Dict, checkpoint_path: str, device: str = 'auto'):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Class information
        self.class_names = config.get('class_names', ['clean', 'dirty'])
        self.clean_idx = 0  # Assuming clean is class 0
        self.dirty_idx = 1  # Assuming dirty is class 1
        
        # Setup paths
        self.output_dir = Path(config.get('checkpoint_dir', './artifacts'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model and data
        self.model = None
        self.data_module = None
        self.temperature = 1.0
        
        self._load_model()
        self._load_data()
        self._load_temperature()
    
    def _load_model(self):
        """Load trained model."""
        print(f"Loading model from {self.checkpoint_path}")
        self.model = load_model(self.checkpoint_path, self.config, str(self.device))
        print(f"Model loaded on {self.device}")
    
    def _load_data(self):
        """Load data module."""
        self.data_module = DirtyCarDataModule(self.config)
        self.data_module.setup()
    
    def _load_temperature(self):
        """Load temperature scaling if available."""
        temp_path = Path(self.checkpoint_path).parent / 'temperature.json'
        if temp_path.exists():
            with open(temp_path, 'r') as f:
                temp_info = json.load(f)
                self.temperature = temp_info.get('temperature', 1.0)
            print(f"Loaded temperature scaling: T = {self.temperature:.4f}")
    
    def collect_predictions(self, dataset_name: str = 'val') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect model predictions on specified dataset.
        
        Args:
            dataset_name: Dataset to use ('train', 'val', 'test')
        
        Returns:
            Tuple of (targets, clean_probabilities, all_probabilities)
        """
        # Get dataloader
        if dataset_name == 'train':
            dataloader = self.data_module.train_dataloader()
        elif dataset_name == 'val':
            dataloader = self.data_module.val_dataloader()
        elif dataset_name == 'test':
            dataloader = self.data_module.test_dataloader()
            if dataloader is None:
                raise ValueError("Test dataset not available")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.model.eval()
        
        all_targets = []
        all_clean_probs = []
        all_probs = []
        
        print(f"Collecting predictions on {dataset_name} dataset...")
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc=f'Processing {dataset_name}'):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                
                # Apply temperature scaling
                scaled_logits = logits / self.temperature
                probabilities = F.softmax(scaled_logits, dim=1)
                
                # Extract clean probabilities (class 0)
                clean_probs = probabilities[:, self.clean_idx]
                
                # Collect results
                all_targets.extend(targets.cpu().numpy())
                all_clean_probs.extend(clean_probs.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        return (
            np.array(all_targets),
            np.array(all_clean_probs),
            np.array(all_probs)
        )
    
    def find_optimal_threshold(
        self,
        targets: np.ndarray,
        clean_probs: np.ndarray,
        target_precision_clean: float = 0.95,
        min_recall_clean: float = 0.1
    ) -> Dict[str, float]:
        """
        Find optimal threshold for target precision on clean class.
        
        Args:
            targets: True labels
            clean_probs: Predicted probabilities for clean class
            target_precision_clean: Target precision for clean class
            min_recall_clean: Minimum acceptable recall for clean class
        
        Returns:
            Dictionary with threshold information
        """
        print(f"Finding optimal threshold for precision_clean >= {target_precision_clean}")
        
        # Generate threshold range
        thresholds = np.linspace(0.01, 0.99, 1000)
        
        precisions = []
        recalls = []
        f1_scores = []
        
        # Calculate metrics for each threshold
        for threshold in thresholds:
            # Business rule: if p_clean >= threshold -> clean, else -> dirty
            predictions = (clean_probs >= threshold).astype(int)
            
            # Calculate precision and recall for clean class (class 0)
            precision_clean = precision_score(targets, predictions, pos_label=0, zero_division=0)
            recall_clean = recall_score(targets, predictions, pos_label=0, zero_division=0)
            
            # F1 score for clean class
            if precision_clean + recall_clean > 0:
                f1_clean = 2 * (precision_clean * recall_clean) / (precision_clean + recall_clean)
            else:
                f1_clean = 0.0
            
            precisions.append(precision_clean)
            recalls.append(recall_clean)
            f1_scores.append(f1_clean)
        
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)
        
        # Find thresholds that meet target precision
        valid_indices = np.where(
            (precisions >= target_precision_clean) & 
            (recalls >= min_recall_clean)
        )[0]
        
        if len(valid_indices) == 0:
            print(f"Warning: No threshold achieves precision >= {target_precision_clean} "
                  f"with recall >= {min_recall_clean}")
            
            # Find best achievable precision
            best_precision_idx = np.argmax(precisions)
            optimal_threshold = thresholds[best_precision_idx]
            achieved_precision = precisions[best_precision_idx]
            achieved_recall = recalls[best_precision_idx]
            achieved_f1 = f1_scores[best_precision_idx]
            
            print(f"Best achievable precision: {achieved_precision:.4f} at threshold {optimal_threshold:.4f}")
        else:
            # Among valid thresholds, choose the one with highest recall (lowest threshold)
            best_valid_idx = valid_indices[np.argmax(recalls[valid_indices])]
            optimal_threshold = thresholds[best_valid_idx]
            achieved_precision = precisions[best_valid_idx]
            achieved_recall = recalls[best_valid_idx]
            achieved_f1 = f1_scores[best_valid_idx]
        
        # Calculate additional metrics at optimal threshold
        optimal_predictions = (clean_probs >= optimal_threshold).astype(int)
        
        # Precision and recall for dirty class
        precision_dirty = precision_score(targets, optimal_predictions, pos_label=1, zero_division=0)
        recall_dirty = recall_score(targets, optimal_predictions, pos_label=1, zero_division=0)
        
        # Overall accuracy
        accuracy = np.mean(targets == optimal_predictions)
        
        # Class distribution analysis
        clean_mask = targets == 0
        dirty_mask = targets == 1
        
        clean_probs_true_clean = clean_probs[clean_mask]
        clean_probs_true_dirty = clean_probs[dirty_mask]
        
        result = {
            'T_clean': float(optimal_threshold),
            'target_precision_clean': float(target_precision_clean),
            'achieved_precision_clean': float(achieved_precision),
            'achieved_recall_clean': float(achieved_recall),
            'achieved_f1_clean': float(achieved_f1),
            'precision_dirty': float(precision_dirty),
            'recall_dirty': float(recall_dirty),
            'overall_accuracy': float(accuracy),
            'temperature': float(self.temperature),
            'min_recall_clean': float(min_recall_clean),
            'num_clean_samples': int(np.sum(clean_mask)),
            'num_dirty_samples': int(np.sum(dirty_mask)),
            'clean_probs_stats': {
                'mean_true_clean': float(np.mean(clean_probs_true_clean)),
                'std_true_clean': float(np.std(clean_probs_true_clean)),
                'mean_true_dirty': float(np.mean(clean_probs_true_dirty)),
                'std_true_dirty': float(np.std(clean_probs_true_dirty))
            }
        }
        
        # Generate visualizations
        self._generate_threshold_visualizations(
            thresholds, precisions, recalls, f1_scores,
            optimal_threshold, target_precision_clean,
            clean_probs_true_clean, clean_probs_true_dirty
        )
        
        return result
    
    def _generate_threshold_visualizations(
        self,
        thresholds: np.ndarray,
        precisions: np.ndarray,
        recalls: np.ndarray,
        f1_scores: np.ndarray,
        optimal_threshold: float,
        target_precision: float,
        clean_probs_true_clean: np.ndarray,
        clean_probs_true_dirty: np.ndarray
    ):
        """Generate threshold analysis visualizations."""
        
        # Threshold analysis plot
        plt.figure(figsize=(15, 10))
        plot_threshold_analysis(
            thresholds, precisions, recalls, f1_scores,
            optimal_threshold, target_precision
        )
        plt.suptitle('Threshold Analysis for Clean Class Precision', fontsize=16)
        plt.savefig(self.output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prediction distribution plot
        plt.figure(figsize=(12, 5))
        plot_prediction_distribution(
            clean_probs_true_clean, clean_probs_true_dirty, optimal_threshold
        )
        plt.suptitle('Distribution of Clean Class Probabilities', fontsize=16)
        plt.savefig(self.output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Threshold analysis visualizations saved")
    
    def analyze_business_impact(self, result: Dict[str, float]) -> Dict[str, any]:
        """Analyze business impact of the selected threshold."""
        
        # Calculate error rates
        false_positive_rate = 1 - result['achieved_recall_dirty']  # Dirty classified as clean
        false_negative_rate = 1 - result['achieved_recall_clean']  # Clean classified as dirty
        
        # Business impact metrics
        total_samples = result['num_clean_samples'] + result['num_dirty_samples']
        
        # Expected errors
        expected_fp = false_positive_rate * result['num_dirty_samples']  # Dirty cars called clean
        expected_fn = false_negative_rate * result['num_clean_samples']  # Clean cars called dirty
        
        business_analysis = {
            'threshold_info': {
                'selected_threshold': result['T_clean'],
                'achieves_target_precision': result['achieved_precision_clean'] >= result['target_precision_clean']
            },
            'error_analysis': {
                'false_positive_rate': float(false_positive_rate),
                'false_negative_rate': float(false_negative_rate),
                'expected_dirty_called_clean': int(expected_fp),
                'expected_clean_called_dirty': int(expected_fn)
            },
            'business_rules': {
                'rule': "if p_clean >= T_clean then 'clean' else 'dirty'",
                'interpretation': f"Cars with clean probability >= {result['T_clean']:.3f} are classified as clean"
            },
            'performance_summary': {
                'clean_precision': result['achieved_precision_clean'],
                'clean_recall': result['achieved_recall_clean'],
                'dirty_precision': result['precision_dirty'],
                'dirty_recall': result['recall_dirty'],
                'overall_accuracy': result['overall_accuracy']
            }
        }
        
        return business_analysis
    
    def run_threshold_selection(
        self,
        dataset_name: str = 'val',
        target_precision_clean: float = 0.95,
        min_recall_clean: float = 0.1,
        output_file: str = 'threshold.json'
    ) -> Dict[str, any]:
        """
        Run complete threshold selection process.
        
        Args:
            dataset_name: Dataset to use for threshold selection
            target_precision_clean: Target precision for clean class
            min_recall_clean: Minimum acceptable recall for clean class
            output_file: Output file name for threshold information
        
        Returns:
            Complete threshold selection results
        """
        print(f"Running threshold selection on {dataset_name} dataset")
        print(f"Target: precision_clean >= {target_precision_clean}")
        print(f"Constraint: recall_clean >= {min_recall_clean}")
        
        # Collect predictions
        targets, clean_probs, all_probs = self.collect_predictions(dataset_name)
        
        print(f"Dataset statistics:")
        print(f"  Total samples: {len(targets)}")
        print(f"  Clean samples: {np.sum(targets == 0)}")
        print(f"  Dirty samples: {np.sum(targets == 1)}")
        print(f"  Class balance: {np.sum(targets == 0) / len(targets):.3f} clean, {np.sum(targets == 1) / len(targets):.3f} dirty")
        
        # Find optimal threshold
        threshold_result = self.find_optimal_threshold(
            targets, clean_probs, target_precision_clean, min_recall_clean
        )
        
        # Analyze business impact
        business_analysis = self.analyze_business_impact(threshold_result)
        
        # Combine results
        complete_result = {
            'threshold_selection': threshold_result,
            'business_analysis': business_analysis,
            'dataset_used': dataset_name,
            'model_info': {
                'checkpoint_path': str(self.checkpoint_path),
                'model_name': self.config.get('model', 'resnet50'),
                'temperature_scaling': self.temperature != 1.0
            }
        }
        
        # Save results
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(complete_result, f, indent=2)
        
        # Print summary
        self._print_summary(threshold_result, business_analysis)
        
        print(f"\nThreshold selection completed!")
        print(f"Results saved to: {output_path}")
        
        return complete_result
    
    def _print_summary(self, threshold_result: Dict, business_analysis: Dict):
        """Print summary of threshold selection results."""
        print(f"\n{'='*60}")
        print("THRESHOLD SELECTION SUMMARY")
        print(f"{'='*60}")
        
        print(f"Selected Threshold: {threshold_result['T_clean']:.4f}")
        print(f"Temperature Scaling: {threshold_result['temperature']:.4f}")
        
        print(f"\nClean Class Performance:")
        print(f"  Target Precision: {threshold_result['target_precision_clean']:.4f}")
        print(f"  Achieved Precision: {threshold_result['achieved_precision_clean']:.4f}")
        print(f"  Achieved Recall: {threshold_result['achieved_recall_clean']:.4f}")
        print(f"  F1 Score: {threshold_result['achieved_f1_clean']:.4f}")
        
        print(f"\nDirty Class Performance:")
        print(f"  Precision: {threshold_result['precision_dirty']:.4f}")
        print(f"  Recall: {threshold_result['recall_dirty']:.4f}")
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {threshold_result['overall_accuracy']:.4f}")
        
        print(f"\nBusiness Impact:")
        error_analysis = business_analysis['error_analysis']
        print(f"  Dirty cars called clean: {error_analysis['expected_dirty_called_clean']}")
        print(f"  Clean cars called dirty: {error_analysis['expected_clean_called_dirty']}")
        print(f"  False positive rate: {error_analysis['false_positive_rate']:.4f}")
        print(f"  False negative rate: {error_analysis['false_negative_rate']:.4f}")
        
        print(f"\nBusiness Rule:")
        print(f"  {business_analysis['business_rules']['interpretation']}")


def main():
    parser = argparse.ArgumentParser(description='Select optimal threshold for DirtyCar classifier')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--probs-source', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset to use for threshold selection')
    parser.add_argument('--target-precision-clean', type=float, default=0.95,
                       help='Target precision for clean class')
    parser.add_argument('--min-recall-clean', type=float, default=0.1,
                       help='Minimum acceptable recall for clean class')
    parser.add_argument('--out', type=str, default='threshold.json',
                       help='Output file name')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create threshold selector
    selector = ThresholdSelector(config, args.checkpoint, args.device)
    
    # Run threshold selection
    results = selector.run_threshold_selection(
        dataset_name=args.probs_source,
        target_precision_clean=args.target_precision_clean,
        min_recall_clean=args.min_recall_clean,
        output_file=args.out
    )


if __name__ == "__main__":
    main()
