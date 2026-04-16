"""
Data Preprocessing and Dataset Preparation for Waterfowl Detection
==================================================================
This script handles:
1. Processing positive images (with birds) and their annotations
2. Processing negative images (without birds) - important for reducing false positives
3. Image quality checks and channel standardization
4. Converting CSV annotations to YOLO format
5. Splitting into train/val/test sets

Why each preprocessing step:
- Negative images: Teach model what's NOT a bird → reduces false positives
- Channel standardization: YOLO requires 3-channel input
- Quality checks: Remove corrupt images that would crash training
- Annotation validation: Ensure bounding boxes are valid
- Dataset balancing: Mix positive/negative samples for better learning
"""

import pandas as pd
from pathlib import Path
import cv2
import shutil
import random
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================================
#                           CONFIGURATION
# ============================================================================

# Source directories (based on your structure)
DATA_DIR = Path("data")
POSITIVE_IMG_DIR = DATA_DIR / "PositiveImage"
NEGATIVE_IMG_DIR = DATA_DIR / "NegativeImage"
LABELS_DIR = DATA_DIR / "PositiveImageLabels"

# Output directories (YOLO format)
OUTPUT_DIR = Path("dataset")
IMG_DIR = OUTPUT_DIR / "images"
LBL_DIR = OUTPUT_DIR / "labels"

# Dataset split ratios
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.1

print("\n" + "="*80)
print("WATERFOWL DETECTION: DATA PREPROCESSING")
print("="*80)

# ============================================================================
#                       PREPROCESSING FUNCTIONS
# ============================================================================

def check_and_standardize_image(img_path):
    """
    Check image quality and standardize to 3-channel BGR.
    
    Why: 
    - Thermal images may have 1 or 4 channels
    - YOLO requires 3-channel input
    - Corrupt images will crash training
    
    Returns: 
    - Standardized image or None if corrupt
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return None
        
        # Convert grayscale (1 channel) to BGR (3 channels)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Convert RGBA (4 channels) to BGR (3 channels)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Already 3 channels
        elif len(img.shape) == 3 and img.shape[2] == 3:
            pass
        
        else:
            print(f" Unexpected image shape: {img.shape} in {img_path.name}")
            return None
        
        return img
        
    except Exception as e:
        print(f" Error reading {img_path.name}: {str(e)}")
        return None


def csv_bbox_to_yolo(x, y, width, height, img_width, img_height):
    """
    Convert CSV bounding box format to YOLO format.
    
    CSV format: (x, y, width, height) - top-left corner + dimensions
    YOLO format: (x_center, y_center, width, height) - all normalized [0,1]
    
    Why: YOLO requires normalized center coordinates
    """
    # Calculate center coordinates
    x_center = (x + width / 2.0) / img_width
    y_center = (y + height / 2.0) / img_height
    
    # Normalize width and height
    norm_width = width / img_width
    norm_height = height / img_height
    
    # Clip to [0, 1] range (safety check)
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    norm_width = np.clip(norm_width, 0.0, 1.0)
    norm_height = np.clip(norm_height, 0.0, 1.0)
    
    return x_center, y_center, norm_width, norm_height


def validate_bbox(x_center, y_center, width, height):
    """
    Validate YOLO format bounding box.
    
    Why: Invalid boxes cause training errors
    """
    return (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
            0 < width <= 1 and 0 < height <= 1)


# ============================================================================
#                       PROCESS POSITIVE IMAGES
# ============================================================================

def process_positive_images():
    """
    Process images with birds and their annotations.
    
    Returns: List of (image_path, yolo_labels) tuples
    """
    print("\n" + "="*80)
    print("STEP 1: Processing Positive Images (with birds)")
    print("="*80)
    
    csv_file = LABELS_DIR / "BoundingBoxLabel.csv"
    
    if not csv_file.exists():
        print(f" CSV file not found: {csv_file}")
        return []
    
    print(f" Reading annotations from: {csv_file.name}")
    df = pd.read_csv(csv_file)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    print(f"   Columns: {list(df.columns)}")
    print(f"   Total annotations: {len(df)}")
    
    # Group by image
    grouped = df.groupby('imageFilename')
    print(f"   Unique images: {len(grouped)}")
    
    positive_items = []
    skipped = 0
    invalid_boxes = 0
    
    print("\n Processing images with annotations...")
    for img_filename, rows in tqdm(grouped, desc="Positive images"):
        img_path = POSITIVE_IMG_DIR / img_filename
        
        if not img_path.exists():
            skipped += 1
            continue
        
        # Check and standardize image
        img = check_and_standardize_image(img_path)
        if img is None:
            skipped += 1
            continue
        
        h, w = img.shape[:2]
        
        # Convert all bounding boxes for this image
        yolo_labels = []
        for _, row in rows.iterrows():
            x = row['x(column)']
            y = row['y(row)']
            width = row['width']
            height = row['height']
            
            # Convert to YOLO format
            x_c, y_c, w_norm, h_norm = csv_bbox_to_yolo(x, y, width, height, w, h)
            
            # Validate
            if validate_bbox(x_c, y_c, w_norm, h_norm):
                # Class 0 = waterfowl
                yolo_labels.append(f"0 {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
            else:
                invalid_boxes += 1
        
        if yolo_labels:
            positive_items.append((img_path, yolo_labels))
    
    print(f"\n Processed positive images:")
    print(f"   Valid images: {len(positive_items)}")
    print(f"   Skipped images: {skipped}")
    print(f"   Invalid boxes: {invalid_boxes}")
    
    return positive_items


# ============================================================================
#                       PROCESS NEGATIVE IMAGES
# ============================================================================

def process_negative_images():
    """
    Process images WITHOUT birds (negative samples).
    
    Why negative images are important:
    - Teach model what background looks like
    - Reduce false positives (warm rocks, vegetation)
    - Improve model's discrimination ability
    - Standard practice in object detection
    
    Returns: List of (image_path, empty_labels) tuples
    """
    print("\n" + "="*80)
    print("STEP 2: Processing Negative Images (without birds)")
    print("="*80)
    print("Why include negatives: Teaches model what's NOT a bird → reduces false alarms")
    
    if not NEGATIVE_IMG_DIR.exists():
        print(f" Negative image directory not found: {NEGATIVE_IMG_DIR}")
        print("  Skipping negative images...")
        return []
    
    negative_items = []
    skipped = 0
    
    neg_images = list(NEGATIVE_IMG_DIR.glob("*"))
    print(f"\n Found {len(neg_images)} negative images")
    
    print(" Processing negative images...")
    for img_path in tqdm(neg_images, desc="Negative images"):
        # Skip non-image files
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            continue
        
        # Check and standardize image
        img = check_and_standardize_image(img_path)
        if img is None:
            skipped += 1
            continue
        
        # Empty label list (no birds in image)
        negative_items.append((img_path, []))
    
    print(f"\n Processed negative images:")
    print(f"   Valid images: {len(negative_items)}")
    print(f"   Skipped images: {skipped}")
    
    return negative_items


# ============================================================================
#                       SPLIT AND SAVE DATASET
# ============================================================================

def split_and_save_dataset(positive_items, negative_items):
    """
    Split dataset into train/val/test and save in YOLO format.
    
    Strategy:
    - Each split contains both positive and negative images
    - Negative images are split proportionally using same ratios as positive
    - This ensures balanced negative distribution across all splits
    """
    print("\n" + "="*80)
    print("STEP 3: Splitting and Saving Dataset")
    print("="*80)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
        (LBL_DIR / split).mkdir(parents=True, exist_ok=True)
    
    # Shuffle both sets
    random.shuffle(positive_items)
    random.shuffle(negative_items)
    
    # Calculate split sizes for positive images
    n_pos = len(positive_items)
    n_pos_train = int(TRAIN_RATIO * n_pos)
    n_pos_val = int(VAL_RATIO * n_pos)
    # Remaining goes to test
    
    # Split negative images using the SAME ratios as positive
    # This ensures balanced distribution across all splits
    n_neg = len(negative_items)
    n_neg_train = int(TRAIN_RATIO * n_neg)
    n_neg_val = int(VAL_RATIO * n_neg)
    n_neg_test = int(TEST_RATIO * n_neg)
    
    # Adjust if total doesn't add up (due to rounding)
    remaining_neg = n_neg - (n_neg_train + n_neg_val + n_neg_test)
    n_neg_train += remaining_neg  # Add any remainder to train
    
    print(f"\n Dataset Split Plan:")
    print(f"{'Split':<10} {'Positive':<12} {'Negative':<12} {'Total':<12} {'Neg %':<10}")
    print("-" * 60)
    
    # Create splits
    splits = {}
    
    # Train split
    pos_train = positive_items[:n_pos_train]
    neg_train = negative_items[:n_neg_train]
    splits['train'] = pos_train + neg_train
    neg_pct_train = len(neg_train) / len(splits['train']) * 100
    print(f"{'Train':<10} {len(pos_train):<12} {len(neg_train):<12} {len(splits['train']):<12} {neg_pct_train:.1f}%")
    
    # Val split
    pos_val = positive_items[n_pos_train:n_pos_train + n_pos_val]
    neg_val = negative_items[n_neg_train:n_neg_train + n_neg_val]
    splits['val'] = pos_val + neg_val
    neg_pct_val = len(neg_val) / len(splits['val']) * 100
    print(f"{'Val':<10} {len(pos_val):<12} {len(neg_val):<12} {len(splits['val']):<12} {neg_pct_val:.1f}%")
    
    # Test split
    pos_test = positive_items[n_pos_train + n_pos_val:]
    neg_test = negative_items[n_neg_train + n_neg_val:]
    splits['test'] = pos_test + neg_test
    neg_pct_test = len(neg_test) / len(splits['test']) * 100 if splits['test'] else 0
    print(f"{'Test':<10} {len(pos_test):<12} {len(neg_test):<12} {len(splits['test']):<12} {neg_pct_test:.1f}%")
    
    # Shuffle each split (mix positive and negative)
    for split_name, items in splits.items():
        random.shuffle(items)
    
    # Save images and labels
    print("\n Saving dataset to disk...")
    for split_name, items in splits.items():
        print(f"\n  Saving {split_name} split...")
        
        for img_path, yolo_labels in tqdm(items, desc=f"  {split_name}"):
            # Read and save image (already preprocessed)
            img = cv2.imread(str(img_path))
            dst_img = IMG_DIR / split_name / img_path.name
            cv2.imwrite(str(dst_img), img)
            
            # Save label file
            dst_lbl = LBL_DIR / split_name / (img_path.stem + '.txt')
            with open(dst_lbl, 'w') as f:
                if yolo_labels:
                    f.write('\n'.join(yolo_labels))
                else:
                    # Empty file for negative images (no birds)
                    pass
    
    print(f"\n Dataset saved to: {OUTPUT_DIR}/")
    
    return splits


# ============================================================================
#                       CREATE YAML CONFIG
# ============================================================================

def create_yaml_config():
    """Create YAML configuration file for YOLO training."""
    print("\n" + "="*80)
    print("STEP 4: Creating YAML Configuration")
    print("="*80)
    
    yaml_content = f"""# Waterfowl Detection Dataset Configuration
# Generated by preprocessing script

path: {OUTPUT_DIR.absolute()}
train: images/train
val: images/val
test: images/test

# Class names
names:
  0: waterfowl

# Number of classes
nc: 1
"""
    
    yaml_path = Path("waterfowl.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"   Configuration saved to: {yaml_path}")
    print(f"   Dataset path: {OUTPUT_DIR.absolute()}")
    print(f"   Classes: waterfowl")


# ============================================================================
#                       GENERATE STATISTICS
# ============================================================================

def print_statistics(positive_items, negative_items):
    """Print dataset statistics."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    # Count total birds
    total_birds = sum(len(labels) for _, labels in positive_items)
    avg_birds = total_birds / len(positive_items) if positive_items else 0
    
    print(f"\n Image Statistics:")
    print(f"   Positive images (with birds): {len(positive_items)}")
    print(f"   Negative images (no birds):   {len(negative_items)}")
    print(f"   Total images:                 {len(positive_items) + len(negative_items)}")
    
    print(f"\n Bird Statistics:")
    print(f"   Total birds annotated: {total_birds}")
    print(f"   Average birds per image: {avg_birds:.2f}")
    
    # Distribution analysis
    birds_per_image = [len(labels) for _, labels in positive_items]
    print(f"\n Distribution:")
    print(f"   Min birds in image: {min(birds_per_image) if birds_per_image else 0}")
    print(f"   Max birds in image: {max(birds_per_image) if birds_per_image else 0}")
    print(f"   Median birds per image: {np.median(birds_per_image) if birds_per_image else 0:.1f}")


# ============================================================================
#                       MAIN EXECUTION
# ============================================================================

def main():
    """Main preprocessing pipeline."""
    
    print("\nPreprocessing Steps:")
    print("1. ✓ Process positive images (with birds) + validate annotations")
    print("2. ✓ Process negative images (without birds) - reduces false positives")
    print("3. ✓ Standardize all images to 3-channel format")
    print("4. ✓ Split into train/val/test (75/15/10) with balanced pos/neg samples")
    print("5. ✓ Convert annotations to YOLO format")
    print("6. ✓ Save dataset and create config file")
    
    # Check if data directories exist
    if not POSITIVE_IMG_DIR.exists():
        print(f"\n Error: Positive image directory not found: {POSITIVE_IMG_DIR}")
        return
    
    # Step 1: Process positive images
    positive_items = process_positive_images()
    
    if not positive_items:
        print("\n No valid positive images found. Cannot proceed.")
        return
    
    # Step 2: Process negative images
    negative_items = process_negative_images()
    
    # Step 3: Print statistics
    print_statistics(positive_items, negative_items)
    
    # Step 4: Split and save
    splits = split_and_save_dataset(positive_items, negative_items)
    
    # Step 5: Create YAML config
    create_yaml_config()
    
    # Final summary
    print("\n" + "="*80)
    print(" PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\n Dataset ready at: {OUTPUT_DIR}/")
    print(f" Config file: waterfowl.yaml")
    print("\n Next step: Train the model")
    print("   Run: python train_yolo.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()