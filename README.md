# YOLOv5 Object Detection for Rock Climbing Holds & Volumes

## Overview
This project trains a YOLOv5 model to detect **rock climbing holds and volumes** from images. The dataset is processed into YOLO format, augmented for class balance, and trained using YOLOv5. Non-Maximum Suppression (NMS) is applied to refine predictions, followed by evaluation against ground truth annotations.

## Directory Structure
```
Identifying-and-Classifying-Climbing-Holds-and-Volumes/
│── yolov5/                          # YOLOv5 model directory
│   ├── train.py                     # Main training script
│   ├── requirements.txt             # Dependencies for YOLOv5
│   ├── runs/                         # Directory for training results and saved weights
│   ├── scripts/                      # Custom scripts for data processing and training
│   │   ├── data_extract.ipynb        # Extracts and preprocesses dataset
│   │   ├── machine_learning.ipynb    # Machine learning model training
│   │   ├── inference.py              # Runs inference on test images
│   │   ├── nms_postprocessing.py     # Applies Non-Maximum Suppression
│
│── yolov5_backup/                    # Backup of YOLOv5 models and training
│
│── data/                             # Raw dataset
│   ├── yolo_data/                    # YOLO-formatted dataset
│   │   ├── train/
│   │   │   ├── images/               # Training images formatted for YOLO
│   │   │   ├── labels/               # Corresponding labels in YOLO format
│   │   ├── valid/
│   │   │   ├── images/               # Validation images formatted for YOLO
│   │   │   ├── labels/               # Corresponding labels in YOLO format
│
│── augmented_data/                    # Augmented dataset for class balancing
│   ├── augmented_images/              # Augmented images
│   ├── augmented_labels/              # Corresponding labels for augmented images
│
│── notebooks/                         # Jupyter notebooks for data processing and training
│   ├── data_extract.ipynb             # Extracts data for training
│   ├── main.ipynb                     # Main training pipeline
│
│── annotations/                        # COCO-format annotation files
│   ├── test_coco_annotations.csv       # Test set annotations
│   ├── train_coco_annotations.csv      # Training set annotations
│   ├── valid_coco_annotations.csv      # Validation set annotations
│
│── .gitignore                          # Ignore unnecessary files (e.g., model weights, cache)
│── README.md                           # Project documentation


```
