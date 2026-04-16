# Waterfowl Detection using YOLOv8
---

## Project Overview
This project implements a YOLOv8-based object detection system to identify waterfowl in images. The pipeline includes data preprocessing, model training, and evaluation with visualizations.

---

## Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n waterfowl python=3.9 -y
conda activate waterfowl
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
ultralytics
torch
torchvision
opencv-python
pandas
numpy
pyyaml
tqdm
matplotlib
```

---

## Project Workflow

### Step 0: Prepare Raw Dataset

**Before running any scripts**, ensure your raw data is organized in the following structure:

```
data/
├── NegativeImage/
├── PositiveImage/
└── PositiveImageLabels/
```

- `NegativeImage/` - Images without waterfowl
- `PositiveImage/` - Images with waterfowl
- `PositiveImageLabels/` - YOLO format label files (.txt) for positive images

---

### Step 1: Data Exploration and Splitting

**Script:** `1_preprocess_and_split.py`

**What it does:**
- Explores the dataset structure and statistics
- Splits data into training (70%), validation (15%), and test (15%) sets
- Creates YOLO-compatible directory structure
- Generates `waterfowl.yaml` configuration file

**Run:**
```bash
python 1_preprocess_and_split.py
```

**Output:**
- `waterfowl_dataset/` folder with organized train/val/test splits
- `waterfowl.yaml` configuration file
- Console output showing dataset statistics

---

### Step 2: Model Training

**Script:** `2_train_yolo.py`

**What it does:**
- Loads pre-trained YOLOv8s model
- Trains on waterfowl dataset for 50 epochs
- Uses Adam optimizer with learning rate 0.001
- Implements early stopping (patience=20)
- Saves training plots and metrics

**Configuration:**
- **Model:** YOLOv8s (small variant)
- **Image size:** 640x640
- **Batch size:** 8
- **Epochs:** 50
- **Device:** GPU (if available) or CPU

**Run:**
```bash
python 2_train_yolo.py
```

**Output:**
- `runs/train/waterfowl_yolo/weights/best.pt` - Best model checkpoint
- `runs/train/waterfowl_yolo/weights/last.pt` - Final model checkpoint
- `runs/train/waterfowl_yolo/results.csv` - Training metrics per epoch
- Training plots (loss curves, mAP, precision, recall)

---

### Step 3: Evaluation and Visualization

**Script:** `3_evaluate_and_visualize.py`

**What it does:**
- Evaluates the trained model on test set
- Computes detection metrics (Precision, Recall, F1-Score)
- Saves 2-3 visual examples for each category:
  - **True Positives (TP):** Correct detections
  - **False Positives (FP):** Incorrect detections
  - **False Negatives (FN):** Missed detections

**Color Coding:**
-  **BLUE** = Ground Truth labels
-  **GREEN** = Correct predictions (TP)
-  **YELLOW** = Incorrect predictions (FP)
-  **RED** = Missed detections (FN)

**Configuration:**
- Confidence threshold: 0.25
- IoU threshold: 0.5

**Run:**
```bash
python 3_evaluate_and_visualize.py
```

**Output:**
- `evaluation_results/` folder containing:
  - `TP_example_1.jpg`, `TP_example_2.jpg`, `TP_example_3.jpg`
  - `FP_example_1.jpg`, `FP_example_2.jpg`, `FP_example_3.jpg`
  - `FN_example_1.jpg`, `FN_example_2.jpg`, `FN_example_3.jpg`
- Console output with detailed metrics

---

## Results Storage

### Directory Structure
```
project/
│
├── waterfowl_dataset/          # Processed dataset
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── runs/train/waterfowl_yolo/  # Training outputs
│   ├── weights/
│   │   ├── best.pt             # Best model
│   │   └── last.pt             # Last epoch model
│   ├── results.csv             # Training metrics
│   └── *.png                   # Training plots
│
└── evaluation_results/          # Evaluation outputs
    ├── TP_example_*.jpg
    ├── FP_example_*.jpg
    └── FN_example_*.jpg
```

### Key Metrics
After evaluation, you'll see:
- **Precision:** Ratio of correct predictions to total predictions
- **Recall:** Ratio of correct predictions to total ground truth objects
- **F1-Score:** Harmonic mean of precision and recall
- **mAP@0.5:** Mean Average Precision at IoU threshold 0.5

---

## Notes

- Ensure your dataset follows YOLO format (images + corresponding .txt label files)
- GPU is recommended for faster training but not required
- Adjust hyperparameters in scripts if needed
- Training time depends on dataset size and hardware (~1-2 hours on GPU)

---

## Troubleshooting

**Issue:** CUDA out of memory  
**Solution:** Reduce `BATCH_SIZE` in `2_train_yolo.py`

**Issue:** Model not found error in evaluation  
**Solution:** Ensure training is completed and `best.pt` exists

**Issue:** Multiprocessing errors on Windows  
**Solution:** Already handled with `workers=0` parameter

---


