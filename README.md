# Helmet Detection System

A real-time helmet detection system using YOLOv8 to identify workers wearing or not wearing helmets in construction site images and videos.

## 🎯 Demo

![Helmet Detection Demo](assets/demo.jpg)

## ✨ Features

- Detects workers with and without safety helmets
- Supports image and video inference
- Built with YOLOv8 and Python
- Designed for construction site safety monitoring

## Project Structure

```
detection_helmet/
├── datasets/
│   └── helmet_yolo/
│       ├── images/        # train / val images
│       ├── labels/        # YOLO format annotations
│       └── data.yaml      # dataset config
├── scripts/
│   ├── train.py                  # Train YOLOv8 model
│   ├── visualize_results.py      # Plot training curves & confusion matrix
│   ├── inference_images.py       # Run detection on test images
│   ├── inference_video.py        # Run detection on video
│   ├── prepare_test_images.py    # Copy val images to test folder
│   └── make_test_video.py        # Create test video from val images
├── runs/
│   └── detect/runs/train/helmet_v1/
│       └── weights/best.pt       # Trained model weights
├── outputs/
│   ├── visualizations/           # Training result charts
│   ├── inference_images/         # Annotated test images
│   └── inference_video/          # Annotated output video
└── README.md
```

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | helmet | Worker wearing a helmet |
| 1 | head | Worker NOT wearing a helmet |
| 2 | person | Person (full body) |

## Requirements

```bash
pip install -r requirements.txt

# Or manually:
pip install ultralytics opencv-python pandas matplotlib torch
```

Recommended:

Python 3.9+
GPU with CUDA support (for faster training)

## Dataset

- **Source**: [Hard Hat Workers Dataset – Roboflow](https://public.roboflow.com/object-detection/hard-hat-workers)
- Format: YOLOv8 (images + YOLO annotation `.txt` files)

**Download and setup:**
1. Download the dataset from the link above (choose **YOLOv8 format**)
2. Place it in the following structure:
```
detection_helmet/
└── datasets/
    └── helmet_yolo/
        ├── images/
│       │   ├── train/
│       │   └── val/
        ├── labels/
        └── data.yaml
```

## How to Run

### 1. Train the model
```bash
python scripts/train.py
```
Trained weights will be saved to `runs/detect/runs/train/helmet_v1/weights/best.pt`

### 2. Visualize training results
```bash
python scripts/visualize_results.py
```
Outputs saved to `outputs/visualizations/`

### 3. Run inference on images
```bash
# First, prepare test images from the val set
python scripts/prepare_test_images.py

# Then run inference
python scripts/inference_images.py
```
Annotated images saved to `outputs/inference_images/`

### 4. Run inference on video
```bash
# Create a test video from val images (if you don't have a video)
python scripts/make_test_video.py

# Then run inference
python scripts/inference_video.py
```
Annotated video saved to `outputs/inference_video/helmet_detection_output.mp4`

## Results

| Metric | Score |
|--------|-------|
| Precision | 0.900 |
| Recall | 0.603 |
| mAP@0.5 | 0.645 |
| mAP@0.5:0.95 | 0.438 |

## Model

- Architecture: YOLOv8n (nano)
- Epochs: 50
- Image size: 640×640
- Device: GPU (CUDA)
