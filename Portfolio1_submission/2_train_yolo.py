"""
YOLOv8 Training Script for Waterfowl Detection
==============================================
"""

from ultralytics import YOLO
import torch
from pathlib import Path

# ============================================================================
#                           CONFIGURATION
# ============================================================================

MODEL_NAME = "yolov8s.pt"
DATA_CONFIG = "waterfowl.yaml"

EPOCHS = 50
BATCH_SIZE = 8
IMG_SIZE = 640
LEARNING_RATE = 0.001
PATIENCE = 20

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

PROJECT = "runs/train"
NAME = "waterfowl_yolo"

# ============================================================================
#                           MAIN TRAINING
# ============================================================================

def main():
    """Main training pipeline."""
    
    print("\n" + "="*80)
    print("WATERFOWL DETECTION - YOLO TRAINING")
    print("="*80)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")
    
    print(f"Model: {MODEL_NAME} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("="*80 + "\n")
    
    # Load model
    model = YOLO(MODEL_NAME)
    
    # Train
    print("Training started...\n")
    
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        optimizer='Adam',
        lr0=LEARNING_RATE,
        patience=PATIENCE,
        device=DEVICE,
        workers=0,  # Fix for Windows multiprocessing
        project=PROJECT,
        name=NAME,
        exist_ok=False,
        plots=True,
        save=True,
        verbose=True,
    )
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    save_dir = Path(PROJECT) / NAME
    print(f"\nModel saved: {save_dir / 'weights' / 'best.pt'}")
    
    # Show final metrics
    results_file = save_dir / 'results.csv'
    if results_file.exists():
        import pandas as pd
        df = pd.read_csv(results_file)
        best_epoch = df['metrics/mAP50(B)'].idxmax()
        
        print("\n" + "-"*80)
        print("FINAL METRICS")
        print("-"*80)
        print(f"Best Epoch: {best_epoch + 1}/{EPOCHS}")
        print(f"mAP@0.5:    {df.loc[best_epoch, 'metrics/mAP50(B)']:.4f}")
        print(f"Precision:  {df.loc[best_epoch, 'metrics/precision(B)']:.4f}")
        print(f"Recall:     {df.loc[best_epoch, 'metrics/recall(B)']:.4f}")
        print("-"*80 + "\n")
    
    print("Next: Run evaluate_and_visualize.py\n")


if __name__ == "__main__":
    main()