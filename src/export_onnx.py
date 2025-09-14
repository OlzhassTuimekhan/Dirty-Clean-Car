"""
ONNX export script for DirtyCar binary classification.
Supports dynamic batch size and FP16 optimization.
"""

import os
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from tqdm import tqdm

# Local imports
from data_module import DirtyCarDataModule, load_config
from model import load_model


class ONNXExporter:
    """ONNX model exporter with optimization and validation."""
    
    def __init__(self, config: Dict, checkpoint_path: str, device: str = 'auto'):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model info
        self.img_size = config.get('img_size', 256)
        self.class_names = config.get('class_names', ['clean', 'dirty'])
        
        # Load model
        self.model = None
        self.temperature = 1.0
        self._load_model()
        self._load_temperature()
    
    def _load_model(self):
        """Load trained model."""
        print(f"Loading model from {self.checkpoint_path}")
        self.model = load_model(self.checkpoint_path, self.config, str(self.device))
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def _load_temperature(self):
        """Load temperature scaling if available."""
        temp_path = Path(self.checkpoint_path).parent / 'temperature.json'
        if temp_path.exists():
            with open(temp_path, 'r') as f:
                temp_info = json.load(f)
                self.temperature = temp_info.get('temperature', 1.0)
            print(f"Loaded temperature scaling: T = {self.temperature:.4f}")
    
    def export_onnx(
        self,
        output_path: str,
        dynamic_batch: bool = True,
        fp16: bool = False,
        opset_version: int = 11,
        simplify: bool = True
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            dynamic_batch: Whether to use dynamic batch size
            fp16: Whether to use FP16 precision
            opset_version: ONNX opset version
            simplify: Whether to simplify the model
        
        Returns:
            Path to exported ONNX model
        """
        print(f"Exporting model to ONNX...")
        print(f"  Output path: {output_path}")
        print(f"  Dynamic batch: {dynamic_batch}")
        print(f"  FP16: {fp16}")
        print(f"  Opset version: {opset_version}")
        
        # Create dummy input
        batch_size = 1
        dummy_input = torch.randn(batch_size, 3, self.img_size, self.img_size, device=self.device)
        
        if fp16:
            self.model.half()
            dummy_input = dummy_input.half()
        
        # Define dynamic axes
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        print(f"Model exported to {output_path}")
        
        # Simplify model if requested
        if simplify:
            try:
                import onnxsim
                print("Simplifying ONNX model...")
                
                # Load and simplify
                model_onnx = onnx.load(output_path)
                model_simplified, check = onnxsim.simplify(model_onnx)
                
                if check:
                    onnx.save(model_simplified, output_path)
                    print("Model simplified successfully")
                else:
                    print("Warning: Model simplification failed, using original model")
            except ImportError:
                print("onnxsim not available, skipping simplification")
        
        return output_path
    
    def validate_onnx_model(
        self,
        onnx_path: str,
        num_samples: int = 16,
        tolerance: float = 1e-3
    ) -> Dict[str, any]:
        """
        Validate ONNX model against PyTorch model.
        
        Args:
            onnx_path: Path to ONNX model
            num_samples: Number of samples to test
            tolerance: Tolerance for numerical differences
        
        Returns:
            Validation results
        """
        print(f"Validating ONNX model with {num_samples} samples...")
        
        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output info
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        
        print(f"ONNX model info:")
        print(f"  Input shape: {input_shape}")
        print(f"  Input name: {input_name}")
        print(f"  Output name: {output_name}")
        
        # Generate test data
        test_inputs = []
        pytorch_outputs = []
        onnx_outputs = []
        
        self.model.eval()
        
        for i in tqdm(range(num_samples), desc="Validating"):
            # Generate random input
            if input_shape[0] == 'batch_size' or input_shape[0] is None:
                # Dynamic batch size
                test_input = torch.randn(1, 3, self.img_size, self.img_size, device=self.device)
            else:
                # Fixed batch size
                test_input = torch.randn(input_shape, device=self.device)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_logits = self.model(test_input)
                
                # Apply temperature scaling
                scaled_logits = pytorch_logits / self.temperature
                pytorch_probs = F.softmax(scaled_logits, dim=1)
                
                pytorch_outputs.append(pytorch_probs.cpu().numpy())
            
            # ONNX inference
            onnx_input = test_input.cpu().numpy()
            onnx_logits = ort_session.run([output_name], {input_name: onnx_input})[0]
            
            # Apply temperature scaling to ONNX output
            onnx_scaled_logits = onnx_logits / self.temperature
            onnx_probs = self._softmax(onnx_scaled_logits)
            
            onnx_outputs.append(onnx_probs)
            test_inputs.append(onnx_input)
        
        # Compare outputs
        pytorch_outputs = np.concatenate(pytorch_outputs, axis=0)
        onnx_outputs = np.concatenate(onnx_outputs, axis=0)
        
        # Calculate differences
        max_diff = np.max(np.abs(pytorch_outputs - onnx_outputs))
        mean_diff = np.mean(np.abs(pytorch_outputs - onnx_outputs))
        
        # Check if within tolerance
        is_valid = max_diff < tolerance
        
        validation_result = {
            'is_valid': bool(is_valid),
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': float(tolerance),
            'num_samples_tested': num_samples,
            'input_shape': input_shape,
            'temperature_applied': self.temperature != 1.0,
            'temperature_value': float(self.temperature)
        }
        
        print(f"Validation results:")
        print(f"  Valid: {is_valid}")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Tolerance: {tolerance}")
        
        if not is_valid:
            print(f"Warning: ONNX model validation failed! Max difference {max_diff:.6f} > tolerance {tolerance}")
        else:
            print("ONNX model validation passed!")
        
        return validation_result
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numpy implementation of softmax."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def benchmark_inference(
        self,
        onnx_path: str,
        num_iterations: int = 100,
        batch_sizes: list = None
    ) -> Dict[str, any]:
        """
        Benchmark ONNX model inference speed.
        
        Args:
            onnx_path: Path to ONNX model
            num_iterations: Number of iterations for benchmarking
            batch_sizes: List of batch sizes to test
        
        Returns:
            Benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        print(f"Benchmarking ONNX model inference...")
        
        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        benchmark_results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size {batch_size}...")
            
            # Generate test data
            test_input = np.random.randn(batch_size, 3, self.img_size, self.img_size).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                _ = ort_session.run([output_name], {input_name: test_input})
            
            # Benchmark
            import time
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = ort_session.run([output_name], {input_name: test_input})
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_iterations
            avg_time_per_sample = avg_time_per_batch / batch_size
            fps = 1.0 / avg_time_per_sample
            
            benchmark_results[f'batch_{batch_size}'] = {
                'avg_time_per_batch_ms': float(avg_time_per_batch * 1000),
                'avg_time_per_sample_ms': float(avg_time_per_sample * 1000),
                'fps': float(fps),
                'batch_size': batch_size
            }
            
            print(f"  Batch {batch_size}: {avg_time_per_batch*1000:.2f}ms/batch, {fps:.1f} FPS")
        
        return benchmark_results
    
    def save_model_metadata(self, onnx_path: str, validation_result: Dict, benchmark_result: Dict = None):
        """Save model metadata and export information."""
        metadata = {
            'model_info': {
                'checkpoint_path': str(self.checkpoint_path),
                'model_name': self.config.get('model', 'resnet50'),
                'num_classes': len(self.class_names),
                'class_names': self.class_names,
                'img_size': self.img_size
            },
            'export_info': {
                'onnx_path': str(onnx_path),
                'temperature_scaling': self.temperature != 1.0,
                'temperature_value': float(self.temperature),
                'export_timestamp': str(torch.datetime.now())
            },
            'validation': validation_result
        }
        
        if benchmark_result:
            metadata['benchmark'] = benchmark_result
        
        # Save metadata
        metadata_path = Path(onnx_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Export DirtyCar model to ONNX')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--out', type=str, default='artifacts/best.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, help='Override image size')
    parser.add_argument('--half', action='store_true', help='Export with FP16 precision')
    parser.add_argument('--dynamic', action='store_true', default=True, help='Use dynamic batch size')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', default=True, help='Simplify ONNX model')
    parser.add_argument('--validate', action='store_true', default=True, help='Validate ONNX model')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark inference speed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override image size if provided
    if args.imgsz:
        config['img_size'] = args.imgsz
    
    # Create exporter
    exporter = ONNXExporter(config, args.checkpoint, args.device)
    
    # Create output directory
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export model
    onnx_path = exporter.export_onnx(
        str(output_path),
        dynamic_batch=args.dynamic,
        fp16=args.half,
        opset_version=args.opset,
        simplify=args.simplify
    )
    
    validation_result = None
    benchmark_result = None
    
    # Validate model
    if args.validate:
        validation_result = exporter.validate_onnx_model(onnx_path)
    
    # Benchmark model
    if args.benchmark:
        benchmark_result = exporter.benchmark_inference(onnx_path)
    
    # Save metadata
    exporter.save_model_metadata(onnx_path, validation_result or {}, benchmark_result)
    
    print(f"\nONNX export completed successfully!")
    print(f"Model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
