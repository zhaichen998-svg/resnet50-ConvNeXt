#!/usr/bin/env python3
"""
YOLOv8 Model Evaluation Script
Calculates mAP, Precision, Recall, and other metrics for object detection models.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Model Evaluation Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the trained model weights (.pt file)'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        required=True,
        help='Path to data configuration YAML file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size for inference'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to run evaluation on (cuda:0, cpu, etc.). Empty string for auto-detect'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.001,
        help='Confidence threshold for predictions'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.6,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save evaluation results to JSON file'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--detailed-metrics',
        action='store_true',
        help='Calculate and display detailed per-class metrics'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed evaluation progress'
    )
    
    return parser.parse_args()


def load_data_config(config_path: str) -> Dict:
    """Load and parse data configuration YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    return device


def load_model(model_path: str, device: torch.device):
    """Load YOLOv8 model from checkpoint."""
    try:
        from ultralytics import YOLO
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        return model
    except ImportError:
        print("Error: ultralytics package not found. Install with: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision using 11-point interpolation.
    """
    # Add sentinel values at the beginning and end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    # Calculate AP using 101-point interpolation
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


class MetricsCalculator:
    """Calculate evaluation metrics for object detection."""
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = {i: [] for i in range(self.num_classes)}
        self.ground_truths = {i: [] for i in range(self.num_classes)}
        self.num_gt_per_class = {i: 0 for i in range(self.num_classes)}
    
    def update(self, predictions: List, ground_truths: List):
        """
        Update metrics with new predictions and ground truths.
        
        Args:
            predictions: List of predicted boxes [class_id, confidence, x1, y1, x2, y2]
            ground_truths: List of ground truth boxes [class_id, x1, y1, x2, y2]
        """
        # Store ground truths
        for gt in ground_truths:
            class_id = int(gt[0])
            self.ground_truths[class_id].append(gt[1:])
            self.num_gt_per_class[class_id] += 1
        
        # Store predictions
        for pred in predictions:
            class_id = int(pred[0])
            confidence = pred[1]
            box = pred[2:]
            self.predictions[class_id].append((confidence, box))
    
    def compute_metrics(self) -> Dict:
        """Compute all metrics including mAP, precision, recall."""
        aps = []
        precisions_per_class = []
        recalls_per_class = []
        
        for class_id in range(self.num_classes):
            if self.num_gt_per_class[class_id] == 0:
                continue
            
            # Sort predictions by confidence (descending)
            preds = sorted(self.predictions[class_id], key=lambda x: x[0], reverse=True)
            
            if len(preds) == 0:
                aps.append(0.0)
                precisions_per_class.append(0.0)
                recalls_per_class.append(0.0)
                continue
            
            # Match predictions to ground truths
            num_preds = len(preds)
            tp = np.zeros(num_preds)
            fp = np.zeros(num_preds)
            
            gt_matched = set()
            
            for pred_idx, (conf, pred_box) in enumerate(preds):
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(self.ground_truths[class_id]):
                    if gt_idx in gt_matched:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= self.iou_threshold:
                    tp[pred_idx] = 1
                    gt_matched.add(best_gt_idx)
                else:
                    fp[pred_idx] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / self.num_gt_per_class[class_id]
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Calculate AP
            ap = calculate_ap(recalls, precisions)
            aps.append(ap)
            
            # Store final precision and recall
            precisions_per_class.append(precisions[-1] if len(precisions) > 0 else 0.0)
            recalls_per_class.append(recalls[-1] if len(recalls) > 0 else 0.0)
        
        # Calculate mean metrics
        mAP = np.mean(aps) if len(aps) > 0 else 0.0
        mean_precision = np.mean(precisions_per_class) if len(precisions_per_class) > 0 else 0.0
        mean_recall = np.mean(recalls_per_class) if len(recalls_per_class) > 0 else 0.0
        
        return {
            'mAP': mAP,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'per_class_ap': aps,
            'per_class_precision': precisions_per_class,
            'per_class_recall': recalls_per_class
        }


def evaluate_model(model, data_config: Dict, args) -> Dict:
    """
    Evaluate model on specified dataset split.
    """
    print(f"\nEvaluating on {args.split} split...")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"IoU threshold: {args.iou_threshold}\n")
    
    # Run validation using ultralytics built-in validator
    try:
        results = model.val(
            data=args.data_config,
            split=args.split,
            imgsz=args.img_size,
            batch=args.batch_size,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            device=args.device,
            workers=args.workers,
            verbose=args.verbose,
            save_json=args.save_json,
            project=args.save_dir,
            name='eval'
        )
        
        # Extract metrics from results
        metrics = {
            'mAP50': float(results.box.map50),  # mAP at IoU=0.5
            'mAP50-95': float(results.box.map),  # mAP at IoU=0.5:0.95
            'precision': float(results.box.mp),  # mean precision
            'recall': float(results.box.mr),     # mean recall
            'num_images': len(results.box.all_ap) if hasattr(results.box, 'all_ap') else 0,
        }
        
        # Add per-class metrics if detailed metrics requested
        if args.detailed_metrics:
            class_names = data_config.get('names', {})
            per_class_metrics = {}
            
            if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'all_ap'):
                for idx, class_idx in enumerate(results.box.ap_class_index):
                    class_name = class_names.get(int(class_idx), f'class_{class_idx}')
                    per_class_metrics[class_name] = {
                        'AP50': float(results.box.all_ap[idx, 0]) if len(results.box.all_ap) > idx else 0.0,
                        'AP50-95': float(np.mean(results.box.all_ap[idx])) if len(results.box.all_ap) > idx else 0.0,
                    }
            
            metrics['per_class_metrics'] = per_class_metrics
        
        return metrics
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_metrics(metrics: Dict, class_names: Optional[Dict] = None):
    """Print evaluation metrics in a formatted table."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Metrics:")
    print(f"  mAP@0.5      : {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics['mAP50-95']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  Images       : {metrics.get('num_images', 'N/A')}")
    
    if 'per_class_metrics' in metrics and metrics['per_class_metrics']:
        print(f"\nPer-Class Metrics:")
        print(f"  {'Class':<20} {'AP@0.5':<10} {'AP@0.5:0.95':<12}")
        print(f"  {'-'*20} {'-'*10} {'-'*12}")
        
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            print(f"  {class_name:<20} {class_metrics['AP50']:<10.4f} {class_metrics['AP50-95']:<12.4f}")
    
    print("\n" + "=" * 70 + "\n")


def save_results(metrics: Dict, args):
    """Save evaluation results to JSON file."""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'model_path': args.model_path,
        'data_config': args.data_config,
        'split': args.split,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'conf_threshold': args.conf_threshold,
        'iou_threshold': args.iou_threshold,
        'metrics': metrics
    }
    
    # Save to JSON
    json_path = save_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {json_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 70)
    print("YOLOv8 Model Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Data Config: {args.data_config}")
    print(f"Split: {args.split}")
    print("=" * 70)
    
    # Load data configuration
    data_config = load_data_config(args.data_config)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Evaluate model
    metrics = evaluate_model(model, data_config, args)
    
    # Print results
    class_names = data_config.get('names', None)
    print_metrics(metrics, class_names)
    
    # Save results if requested
    if args.save_json:
        save_results(metrics, args)
    
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
