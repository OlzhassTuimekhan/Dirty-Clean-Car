"""
Evaluation script for DirtyCar binary classification.
Generates comprehensive metrics, confusion matrix, and reports.
"""

import os
import argparse
import yaml
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from tqdm import tqdm

# Local imports
from data_module import DirtyCarDataModule, load_config
from model import create_model, load_model
from utils_vis import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve


class ModelEvaluator:
    """Comprehensive model evaluation for DirtyCar classifier."""
    
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
        self.num_classes = len(self.class_names)
        
        # Setup paths
        self.output_dir = Path(config.get('checkpoint_dir', './artifacts')) / 'reports'
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
    
    def evaluate_dataset(self, dataset_name: str = 'val') -> Dict[str, any]:
        """
        Evaluate model on specified dataset.
        
        Args:
            dataset_name: 'train', 'val', or 'test'
        
        Returns:
            Dictionary with predictions, probabilities, and metrics
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
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_logits = []
        
        print(f"Evaluating on {dataset_name} dataset...")
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc=f'Evaluating {dataset_name}'):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                
                # Apply temperature scaling
                scaled_logits = logits / self.temperature
                probabilities = F.softmax(scaled_logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        targets = np.array(all_targets)
        logits = np.array(all_logits)
        
        # Calculate metrics
        metrics = self._calculate_metrics(targets, predictions, probabilities)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'targets': targets,
            'logits': logits,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, targets: np.ndarray, predictions: np.ndarray, 
                          probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        # ROC AUC and AP scores
        try:
            if self.num_classes == 2:
                # Binary classification
                roc_auc = roc_auc_score(targets, probabilities[:, 1])
                ap_score = average_precision_score(targets, probabilities[:, 1])
            else:
                # Multi-class
                roc_auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
                ap_score = average_precision_score(targets, probabilities, average='macro')
        except ValueError:
            roc_auc = 0.0
            ap_score = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Per-class metrics
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'roc_auc': roc_auc,
            'average_precision': ap_score,
            'confusion_matrix': cm.tolist(),
            'support': support.tolist()
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision[i] if i < len(precision) else 0.0
            metrics[f'{class_name}_recall'] = recall[i] if i < len(recall) else 0.0
            metrics[f'{class_name}_f1'] = f1[i] if i < len(f1) else 0.0
            metrics[f'{class_name}_support'] = int(support[i]) if i < len(support) else 0
        
        return metrics
    
    def generate_classification_report(self, results: Dict, dataset_name: str):
        """Generate detailed classification report."""
        targets = results['targets']
        predictions = results['predictions']
        
        # Scikit-learn classification report
        report = classification_report(
            targets, predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save as JSON
        report_path = self.output_dir / f'{dataset_name}_classification_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as text
        report_text = classification_report(
            targets, predictions,
            target_names=self.class_names
        )
        
        report_text_path = self.output_dir / f'{dataset_name}_classification_report.txt'
        with open(report_text_path, 'w') as f:
            f.write(report_text)
        
        print(f"Classification report saved to {report_path}")
        return report
    
    def save_predictions(self, results: Dict, dataset_name: str):
        """Save predictions to CSV file."""
        predictions_path = self.output_dir / f'{dataset_name}_predictions.csv'
        
        with open(predictions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['sample_id', 'true_label', 'predicted_label', 'true_class', 'predicted_class']
            header.extend([f'prob_{class_name}' for class_name in self.class_names])
            writer.writerow(header)
            
            # Data
            for i, (target, pred) in enumerate(zip(results['targets'], results['predictions'])):
                row = [
                    i,
                    int(target),
                    int(pred),
                    self.class_names[target],
                    self.class_names[pred]
                ]
                row.extend([f"{prob:.6f}" for prob in results['probabilities'][i]])
                writer.writerow(row)
        
        print(f"Predictions saved to {predictions_path}")
    
    def save_metrics_summary(self, all_results: Dict[str, Dict]):
        """Save comprehensive metrics summary."""
        summary = {
            'model_info': {
                'checkpoint_path': str(self.checkpoint_path),
                'model_name': self.config.get('model', 'resnet50'),
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'temperature': self.temperature
            },
            'datasets': {}
        }
        
        for dataset_name, results in all_results.items():
            summary['datasets'][dataset_name] = results['metrics']
        
        # Save summary
        summary_path = self.output_dir / 'evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create CSV summary
        csv_path = self.output_dir / 'metrics_summary.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['dataset', 'accuracy', 'macro_f1', 'weighted_f1', 'roc_auc', 'average_precision']
            header.extend([f'{class_name}_f1' for class_name in self.class_names])
            writer.writerow(header)
            
            # Data
            for dataset_name, results in all_results.items():
                metrics = results['metrics']
                row = [
                    dataset_name,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                    f"{metrics['weighted_f1']:.4f}",
                    f"{metrics['roc_auc']:.4f}",
                    f"{metrics['average_precision']:.4f}"
                ]
                row.extend([f"{metrics.get(f'{class_name}_f1', 0.0):.4f}" for class_name in self.class_names])
                writer.writerow(row)
        
        print(f"Metrics summary saved to {summary_path}")
        print(f"CSV summary saved to {csv_path}")
    
    def run_full_evaluation(self, datasets: List[str] = None) -> Dict[str, Dict]:
        """Run comprehensive evaluation on specified datasets."""
        if datasets is None:
            datasets = ['val']
            # Add test if available
            if self.data_module.test_dataloader() is not None:
                datasets.append('test')
        
        all_results = {}
        
        for dataset_name in datasets:
            print(f"\n{'='*50}")
            print(f"Evaluating on {dataset_name.upper()} dataset")
            print(f"{'='*50}")
            
            try:
                # Evaluate
                results = self.evaluate_dataset(dataset_name)
                all_results[dataset_name] = results
                
                # Print metrics
                metrics = results['metrics']
                print(f"\nMetrics for {dataset_name}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Macro F1: {metrics['macro_f1']:.4f}")
                print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
                print(f"  Average Precision: {metrics['average_precision']:.4f}")
                
                # Per-class metrics
                for class_name in self.class_names:
                    f1 = metrics.get(f'{class_name}_f1', 0.0)
                    precision = metrics.get(f'{class_name}_precision', 0.0)
                    recall = metrics.get(f'{class_name}_recall', 0.0)
                    support = metrics.get(f'{class_name}_support', 0)
                    print(f"  {class_name.capitalize()}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, Support={support}")
                
                # Generate reports
                self.generate_classification_report(results, dataset_name)
                self.save_predictions(results, dataset_name)
                
                # Generate visualizations
                self._generate_visualizations(results, dataset_name)
                
            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                continue
        
        # Save comprehensive summary
        if all_results:
            self.save_metrics_summary(all_results)
        
        return all_results
    
    def _generate_visualizations(self, results: Dict, dataset_name: str):
        """Generate visualization plots."""
        targets = results['targets']
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(cm, self.class_names, normalize=True)
        plt.title(f'Confusion Matrix - {dataset_name.upper()}')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # For binary classification, generate additional plots
        if self.num_classes == 2:
            # ROC curve
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            plt.figure(figsize=(8, 6))
            plot_roc_curve(fpr, tpr, results['metrics']['roc_auc'])
            plt.title(f'ROC Curve - {dataset_name.upper()}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{dataset_name}_roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(targets, probabilities[:, 1])
            plt.figure(figsize=(8, 6))
            plot_precision_recall_curve(recall_curve, precision_curve, results['metrics']['average_precision'])
            plt.title(f'Precision-Recall Curve - {dataset_name.upper()}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{dataset_name}_pr_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved for {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DirtyCar classifier')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, help='Override data root path')
    parser.add_argument('--datasets', nargs='+', default=['val'], 
                       choices=['train', 'val', 'test'], help='Datasets to evaluate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override data root if provided
    if args.data_root:
        config['data_root'] = args.data_root
    
    # Create evaluator
    evaluator = ModelEvaluator(config, args.checkpoint, args.device)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(args.datasets)
    
    print(f"\nEvaluation completed! Results saved to {evaluator.output_dir}")


if __name__ == "__main__":
    main()
