"""
Model Evaluation and Visualization for Waterfowl Detection
===========================================================
This script:
1. Evaluates model on test set
2. Computes metrics (Precision, Recall, F1, mAP)
3. Saves ONLY 2-3 examples each of:
   - True Positives (correct detections)
   - False Positives (incorrect detections)
   - False Negatives (missed detections)

Color Coding:
- BLUE boxes = Ground Truth labels
- GREEN boxes = Correct Predictions (TP)
- YELLOW boxes = Incorrect Predictions (FP)
- RED boxes = Missed Ground Truth (FN)

Shows: Confidence scores only (no class names)
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import yaml
import random
from tqdm import tqdm

random.seed(42)

# ============================================================================
#                           CONFIGURATION
# ============================================================================

MODEL_PATH = "runs/train/waterfowl_yolo/weights/best.pt"
DATA_CONFIG = "waterfowl.yaml"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# Output directory - CLEAN and organized
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Color scheme (BGR format)
COLORS = {
    'GT': (255, 0, 0),      # BLUE - Ground Truth
    'TP': (0, 255, 0),      # GREEN - True Positive (correct)
    'FP': (0, 255, 255),    # YELLOW - False Positive (wrong)
    'FN': (0, 0, 255),      # RED - False Negative (missed)
}

EXAMPLES_TO_SAVE = 3  # Save 2-3 examples per category

# ============================================================================
#                           HELPER FUNCTIONS
# ============================================================================

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def parse_yolo_label(label_path, img_width, img_height):
    """Parse YOLO label file and return boxes in [x1, y1, x2, y2] format."""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            _, xc, yc, w, h = map(float, parts)
            
            x1 = (xc - w / 2) * img_width
            y1 = (yc - h / 2) * img_height
            x2 = (xc + w / 2) * img_width
            y2 = (yc + h / 2) * img_height
            
            boxes.append([x1, y1, x2, y2])
    
    return boxes


def draw_box(img, box, color, label="", thickness=2):
    """Draw a single box with optional confidence label."""
    x1, y1, x2, y2 = map(int, box[:4])
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label with confidence
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Background for text
        cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        
        # Text
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, 
                    (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    return img


# ============================================================================
#                           EVALUATION
# ============================================================================

def evaluate_model():
    """Evaluate model and collect TP/FP/FN examples."""
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Load dataset config
    with open(DATA_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    test_images_dir = dataset_path / 'images' / 'test'
    test_labels_dir = dataset_path / 'labels' / 'test'
    
    test_images = list(test_images_dir.glob("*"))
    print(f"Test images: {len(test_images)}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}\n")
    
    # Metrics
    metrics = {
        'TP': 0, 'FP': 0, 'FN': 0,
        'tp_examples': [],
        'fp_examples': [],
        'fn_examples': []
    }
    
    print("Running inference on test set...")
    
    for img_path in tqdm(test_images):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Get ground truth
        label_path = test_labels_dir / (img_path.stem + '.txt')
        gt_boxes = parse_yolo_label(label_path, w, h)
        
        # Get predictions
        results = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        
        pred_boxes = []
        pred_confs = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                pred_boxes.append(box.xyxy[0].cpu().numpy().tolist())
                pred_confs.append(box.conf[0].cpu().numpy().item())
        
        # Match predictions with ground truth
        matched_gt = set()
        matched_pred = set()
        tp_pairs = []
        
        for pred_idx, (pred_box, conf) in enumerate(zip(pred_boxes, pred_confs)):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= IOU_THRESHOLD:
                # True Positive
                metrics['TP'] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                tp_pairs.append((pred_box, conf, gt_boxes[best_gt_idx]))
        
        # False Positives
        fp_boxes = [(pred_boxes[i], pred_confs[i]) for i in range(len(pred_boxes)) 
                    if i not in matched_pred]
        metrics['FP'] += len(fp_boxes)
        
        # False Negatives
        fn_boxes = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
        metrics['FN'] += len(fn_boxes)
        
        # Store examples
        if tp_pairs and len(metrics['tp_examples']) < EXAMPLES_TO_SAVE:
            metrics['tp_examples'].append({
                'img_path': img_path,
                'tp_pairs': tp_pairs,
                'all_gt': gt_boxes
            })
        
        if fp_boxes and len(metrics['fp_examples']) < EXAMPLES_TO_SAVE:
            metrics['fp_examples'].append({
                'img_path': img_path,
                'fp_boxes': fp_boxes,
                'all_gt': gt_boxes
            })
        
        if fn_boxes and len(metrics['fn_examples']) < EXAMPLES_TO_SAVE:
            metrics['fn_examples'].append({
                'img_path': img_path,
                'fn_boxes': fn_boxes,
                'all_pred': [(pred_boxes[i], pred_confs[i]) for i in range(len(pred_boxes))]
            })
    
    return metrics


# ============================================================================
#                           VISUALIZATION
# ============================================================================

def save_visualizations(metrics):
    """Save visualization examples for TP, FP, FN."""
    
    print("\n" + "="*80)
    print("SAVING VISUALIZATIONS")
    print("="*80)
    
    # True Positives
    if metrics['tp_examples']:
        print(f"\nSaving {len(metrics['tp_examples'])} True Positive examples...")
        for i, example in enumerate(metrics['tp_examples'][:EXAMPLES_TO_SAVE], 1):
            img = cv2.imread(str(example['img_path']))
            
            # Draw all ground truth in BLUE (thin)
            for gt_box in example['all_gt']:
                draw_box(img, gt_box, COLORS['GT'], "", thickness=1)
            
            # Draw correct predictions in GREEN with confidence
            for pred_box, conf, _ in example['tp_pairs']:
                label = f"{conf:.2f}"
                draw_box(img, pred_box, COLORS['TP'], label, thickness=2)
            
            output_path = OUTPUT_DIR / f"TP_example_{i}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"  ✓ {output_path.name}")
    
    # False Positives
    if metrics['fp_examples']:
        print(f"\nSaving {len(metrics['fp_examples'])} False Positive examples...")
        for i, example in enumerate(metrics['fp_examples'][:EXAMPLES_TO_SAVE], 1):
            img = cv2.imread(str(example['img_path']))
            
            # Draw ground truth in BLUE
            for gt_box in example['all_gt']:
                draw_box(img, gt_box, COLORS['GT'], "GT", thickness=2)
            
            # Draw false positives in YELLOW with confidence
            for fp_box, conf in example['fp_boxes']:
                label = f"{conf:.2f}"
                draw_box(img, fp_box, COLORS['FP'], label, thickness=2)
            
            output_path = OUTPUT_DIR / f"FP_example_{i}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"  ✓ {output_path.name}")
    
    # False Negatives
    if metrics['fn_examples']:
        print(f"\nSaving {len(metrics['fn_examples'])} False Negative examples...")
        for i, example in enumerate(metrics['fn_examples'][:EXAMPLES_TO_SAVE], 1):
            img = cv2.imread(str(example['img_path']))
            
            # Draw missed ground truth in RED
            for fn_box in example['fn_boxes']:
                draw_box(img, fn_box, COLORS['FN'], "Missed", thickness=2)
            
            # Draw existing predictions in GREEN (thin)
            for pred_box, conf in example['all_pred']:
                label = f"{conf:.2f}"
                draw_box(img, pred_box, COLORS['TP'], label, thickness=1)
            
            output_path = OUTPUT_DIR / f"FN_example_{i}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"  ✓ {output_path.name}")


# ============================================================================
#                           RESULTS
# ============================================================================

def print_metrics(metrics):
    """Print evaluation metrics."""
    
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    tp = metrics['TP']
    fp = metrics['FP']
    fn = metrics['FN']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetection Statistics:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COLOR CODE")
    print("="*80)
    print("  BLUE   = Ground Truth")
    print("  GREEN  = Correct Predictions (TP)")
    print("  YELLOW = Wrong Predictions (FP)")
    print("  RED    = Missed Ground Truth (FN)")
    print("="*80)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("  - TP_example_1.jpg, TP_example_2.jpg, TP_example_3.jpg")
    print("  - FP_example_1.jpg, FP_example_2.jpg, FP_example_3.jpg")
    print("  - FN_example_1.jpg, FN_example_2.jpg, FN_example_3.jpg")
    print("\n")


# ============================================================================
#                           MAIN
# ============================================================================

def main():
    """Main evaluation pipeline."""
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n Model not found: {MODEL_PATH}")
        print("Please train the model first!\n")
        return
    
    # Evaluate
    metrics = evaluate_model()
    
    # Print metrics
    print_metrics(metrics)
    
    # Save visualizations
    save_visualizations(metrics)
    
    print("Evaluation complete!\n")


if __name__ == "__main__":
    main()