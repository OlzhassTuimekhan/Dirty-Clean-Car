"""
Batch inference utility for DirtyCar classification.
Process multiple images from directory or CSV file.
"""

import os
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import time

import numpy as np
import cv2
from tqdm import tqdm
import onnxruntime as ort

# Local imports
from serve import DirtyCarPredictor


class BatchInferenceProcessor:
    """Batch processing for DirtyCar classification."""
    
    def __init__(self, model_path: str, img_size: int = 256):
        """
        Initialize batch processor.
        
        Args:
            model_path: Path to ONNX model
            img_size: Input image size
        """
        # Set environment variables for predictor
        os.environ['MODEL_PATH'] = model_path
        os.environ['IMGSZ'] = str(img_size)
        
        # Initialize predictor
        self.predictor = DirtyCarPredictor()
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    
    def process_directory(
        self,
        input_dir: str,
        output_csv: str,
        recursive: bool = True,
        batch_size: int = 32
    ) -> Dict[str, any]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_csv: Output CSV file path
            recursive: Whether to search recursively
            batch_size: Batch size for processing
        
        Returns:
            Processing statistics
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all image files
        image_files = []
        pattern = "**/*" if recursive else "*"
        
        for ext in self.image_extensions:
            image_files.extend(input_path.glob(f"{pattern}{ext}"))
            image_files.extend(input_path.glob(f"{pattern}{ext.upper()}"))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        if len(image_files) == 0:
            print("No images found!")
            return {'total_images': 0, 'processed': 0, 'errors': 0}
        
        # Process images
        results = []
        errors = []
        
        start_time = time.time()
        
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    errors.append({'path': str(image_path), 'error': 'Failed to load image'})
                    continue
                
                # Predict
                result = self.predictor.predict_image(image)
                
                # Add file information
                result.update({
                    'file_path': str(image_path),
                    'file_name': image_path.name,
                    'relative_path': str(image_path.relative_to(input_path))
                })
                
                results.append(result)
                
            except Exception as e:
                errors.append({'path': str(image_path), 'error': str(e)})
        
        processing_time = time.time() - start_time
        
        # Save results to CSV
        self._save_results_csv(results, output_csv)
        
        # Print statistics
        stats = {
            'total_images': len(image_files),
            'processed': len(results),
            'errors': len(errors),
            'processing_time_seconds': processing_time,
            'images_per_second': len(results) / processing_time if processing_time > 0 else 0
        }
        
        print(f"\nProcessing completed:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Successfully processed: {stats['processed']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Processing time: {stats['processing_time_seconds']:.2f} seconds")
        print(f"  Speed: {stats['images_per_second']:.1f} images/second")
        
        # Save error log if there are errors
        if errors:
            error_log_path = Path(output_csv).with_suffix('.errors.json')
            with open(error_log_path, 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"Error log saved to: {error_log_path}")
        
        return stats
    
    def process_csv_list(
        self,
        input_csv: str,
        output_csv: str,
        image_path_column: str = 'image_path',
        base_path: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Process images listed in a CSV file.
        
        Args:
            input_csv: Input CSV file with image paths
            output_csv: Output CSV file path
            image_path_column: Column name containing image paths
            base_path: Base path to prepend to relative paths
        
        Returns:
            Processing statistics
        """
        input_path = Path(input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        
        # Read CSV
        image_paths = []
        additional_data = []
        
        with open(input_csv, 'r') as f:
            reader = csv.DictReader(f)
            
            if image_path_column not in reader.fieldnames:
                raise ValueError(f"Column '{image_path_column}' not found in CSV")
            
            for row in reader:
                img_path = row[image_path_column]
                
                # Handle relative paths
                if base_path and not Path(img_path).is_absolute():
                    img_path = str(Path(base_path) / img_path)
                
                image_paths.append(img_path)
                additional_data.append(row)
        
        print(f"Found {len(image_paths)} images in CSV")
        
        # Process images
        results = []
        errors = []
        
        start_time = time.time()
        
        for i, (image_path, extra_data) in enumerate(tqdm(
            zip(image_paths, additional_data), 
            total=len(image_paths),
            desc="Processing images"
        )):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    errors.append({'path': image_path, 'error': 'Failed to load image', 'row': i})
                    continue
                
                # Predict
                result = self.predictor.predict_image(image)
                
                # Add file and extra information
                result.update({
                    'file_path': image_path,
                    'file_name': Path(image_path).name,
                    'csv_row': i
                })
                
                # Add additional data from CSV
                result.update(extra_data)
                
                results.append(result)
                
            except Exception as e:
                errors.append({'path': image_path, 'error': str(e), 'row': i})
        
        processing_time = time.time() - start_time
        
        # Save results
        self._save_results_csv(results, output_csv)
        
        # Statistics
        stats = {
            'total_images': len(image_paths),
            'processed': len(results),
            'errors': len(errors),
            'processing_time_seconds': processing_time,
            'images_per_second': len(results) / processing_time if processing_time > 0 else 0
        }
        
        print(f"\nProcessing completed:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Successfully processed: {stats['processed']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Processing time: {stats['processing_time_seconds']:.2f} seconds")
        print(f"  Speed: {stats['images_per_second']:.1f} images/second")
        
        # Save error log
        if errors:
            error_log_path = Path(output_csv).with_suffix('.errors.json')
            with open(error_log_path, 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"Error log saved to: {error_log_path}")
        
        return stats
    
    def _save_results_csv(self, results: List[Dict], output_path: str):
        """Save results to CSV file."""
        if not results:
            print("No results to save")
            return
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Get all possible fieldnames
        fieldnames = set()
        for result in results:
            fieldnames.update(result.keys())
        
        # Sort fieldnames for consistent output
        fieldnames = sorted(fieldnames)
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to: {output_path}")
    
    def generate_summary_report(self, results_csv: str, output_dir: str = None):
        """Generate summary report from results CSV."""
        if output_dir is None:
            output_dir = Path(results_csv).parent
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Read results
        results = []
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        if not results:
            print("No results found in CSV")
            return
        
        # Calculate statistics
        total_images = len(results)
        clean_predictions = sum(1 for r in results if r['label'] == 'clean')
        dirty_predictions = sum(1 for r in results if r['label'] == 'dirty')
        unsure_predictions = sum(1 for r in results if r['label'] == 'unsure')
        
        # Confidence distribution
        high_confidence = sum(1 for r in results if r.get('confidence') == 'high')
        medium_confidence = sum(1 for r in results if r.get('confidence') == 'medium')
        low_confidence = sum(1 for r in results if r.get('confidence') == 'low')
        
        # Probability statistics
        p_clean_values = [float(r['p_clean']) for r in results]
        avg_p_clean = np.mean(p_clean_values)
        std_p_clean = np.std(p_clean_values)
        
        # Processing time statistics
        if 'processing_time_ms' in results[0]:
            processing_times = [float(r['processing_time_ms']) for r in results]
            avg_processing_time = np.mean(processing_times)
            total_processing_time = sum(processing_times)
        else:
            avg_processing_time = None
            total_processing_time = None
        
        # Create summary
        summary = {
            'total_images': total_images,
            'predictions': {
                'clean': clean_predictions,
                'dirty': dirty_predictions,
                'unsure': unsure_predictions
            },
            'prediction_percentages': {
                'clean': clean_predictions / total_images * 100,
                'dirty': dirty_predictions / total_images * 100,
                'unsure': unsure_predictions / total_images * 100
            },
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            },
            'probability_statistics': {
                'mean_p_clean': avg_p_clean,
                'std_p_clean': std_p_clean,
                'min_p_clean': min(p_clean_values),
                'max_p_clean': max(p_clean_values)
            }
        }
        
        if avg_processing_time is not None:
            summary['performance'] = {
                'avg_processing_time_ms': avg_processing_time,
                'total_processing_time_ms': total_processing_time,
                'images_per_second': 1000 / avg_processing_time
            }
        
        # Save summary
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\nBatch Processing Summary:")
        print(f"  Total images: {total_images}")
        print(f"  Clean: {clean_predictions} ({clean_predictions/total_images*100:.1f}%)")
        print(f"  Dirty: {dirty_predictions} ({dirty_predictions/total_images*100:.1f}%)")
        if unsure_predictions > 0:
            print(f"  Unsure: {unsure_predictions} ({unsure_predictions/total_images*100:.1f}%)")
        print(f"  Average p_clean: {avg_p_clean:.3f} ± {std_p_clean:.3f}")
        
        if avg_processing_time is not None:
            print(f"  Average processing time: {avg_processing_time:.1f} ms")
            print(f"  Processing speed: {1000/avg_processing_time:.1f} images/second")
        
        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch inference for DirtyCar classifier')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input directory or CSV file')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output CSV file')
    parser.add_argument('--imgsz', type=int, default=256, help='Input image size')
    parser.add_argument('--recursive', action='store_true', 
                       help='Search directories recursively')
    parser.add_argument('--csv-column', type=str, default='image_path',
                       help='CSV column name for image paths')
    parser.add_argument('--base-path', type=str,
                       help='Base path for relative image paths in CSV')
    parser.add_argument('--summary', action='store_true',
                       help='Generate summary report')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchInferenceProcessor(args.model, args.imgsz)
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Process directory
        stats = processor.process_directory(
            args.input, 
            args.output, 
            recursive=args.recursive
        )
    elif input_path.suffix.lower() == '.csv':
        # Process CSV
        stats = processor.process_csv_list(
            args.input,
            args.output,
            image_path_column=args.csv_column,
            base_path=args.base_path
        )
    else:
        raise ValueError("Input must be a directory or CSV file")
    
    # Generate summary if requested
    if args.summary:
        processor.generate_summary_report(args.output)


if __name__ == "__main__":
    main()
