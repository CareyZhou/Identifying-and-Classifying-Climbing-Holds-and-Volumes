# YOLOv5 Object Detection for Rock Climbing Holds & Volumes

## Overview
This project trains a YOLOv5 model to detect **rock climbing holds and volumes** from images. The dataset is processed into YOLO format, augmented for class balance, and trained using YOLOv5. Non-Maximum Suppression (NMS) is applied to refine predictions, followed by evaluation against ground truth annotations.

## Directory Structure
```
project_root/
│── yolov5/                          # YOLOv5 model directory
│   ├── train.py                     # Main training script
│   ├── requirements.txt             # Dependencies for YOLOv5
│   ├── runs/                         # Directory for training results and saved weights
│   ├── scripts/                      # Custom scripts for data processing and training
│   │   ├── data_preparation.py       # Converts dataset to YOLO format
│   │   ├── augmentation.py           # Applies data augmentation
│   │   ├── train_model.py            # Trains YOLO model
│   │   ├── inference.py              # Runs inference on test images
│   │   ├── nms_postprocessing.py     # Applies Non-Maximum Suppression
│   ├── runs/hold_volume_detection8/weights/best.pt  # Trained model weights
│
│── data/                             # Raw dataset
│   ├── train/                        # Original training images
│   ├── valid/                        # Original validation images
│   ├── test_image/                    # Test images for inference
│   ├── test_output/                   # YOLO model predictions output
│
│── yolo_data/                        # YOLO-formatted dataset
│   ├── train/
│   │   ├── images/                    # Training images formatted for YOLO
│   │   ├── labels/                    # Corresponding labels in YOLO format
│   ├── valid/
│   │   ├── images/                    # Validation images formatted for YOLO
│   │   ├── labels/                    # Corresponding labels in YOLO format
│
│── augmented_data/                    # Augmented dataset for class balancing
│   ├── images/                         # Augmented images
│   ├── labels/                         # Corresponding labels for augmented images
│
│── README.md                          # Project documentation

```

## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/your-repo/yolo-climbing-detection.git
cd yolo-climbing-detection
```
### 2. Install Dependencies
```sh
pip install -r yolov5/requirements.txt
pip install numpy pandas tensorflow keras opencv-python scikit-learn matplotlib torch torchvision tqdm imgaug albumentations
```

## Dataset Preparation
### 1. Convert COCO Annotations to YOLO Format
Run the script to process COCO-style annotations into YOLO format:
```sh
python scripts/data_preparation.py
```
### 2. Apply Data Augmentation
To balance the dataset, augment images containing **volumes**:
```sh
python scripts/augmentation.py
```

## Training YOLOv5
To train the YOLO model with the prepared dataset:
```sh
python yolov5/train.py --img 416 --batch 8 --epochs 30 --data yolo_data/data.yaml --weights yolov5s.pt --project runs --name hold_volume_detection --workers 4
```
To resume a previous training run:
```sh
python yolov5/train.py --resume
```

## Running Inference
Use the trained model to make predictions on test images:
```sh
python scripts/inference.py
```

## Post-Processing
### 1. Apply Non-Maximum Suppression (NMS)
Standard NMS:
```sh
python scripts/nms_postprocessing.py --input test_statistics.csv --output nms_statistics.csv
```
Improved NMS with confidence weighting:
```sh
python scripts/nms_postprocessing.py --input test_statistics.csv --output nms_statistics_improved.csv --improved
```
### 2. Validate Predictions
Compare predictions against ground truth:
```sh
python scripts/validate_results.py
```

## Evaluation
To evaluate the performance of YOLOv5 on validation images:
```sh
python scripts/evaluate.py
```

## Results & Visualization
1. **Bounding Box Visualization:**
   - Standard predictions are saved in `data/test_output/`
   - Post-NMS results are in `data/nms_output/`
   - Mismatched predictions (NMS vs. weighted NMS) are saved in `data/mismatched_nms_output/`

2. **Class Distribution Analysis:**
   - Check class balance after oversampling:
   ```sh
   python scripts/check_class_distribution.py
   ```

## Future Work
- **Hyperparameter tuning**: Experiment with `hyp.yaml` values for better performance.
- **Model fine-tuning**: Try `yolov5m.pt` or `yolov5l.pt` for better accuracy.
- **Edge case handling**: Improve detection for difficult lighting conditions or occlusions.

## Acknowledgments
- YOLOv5 by Ultralytics: https://github.com/ultralytics/yolov5
- Image Augmentation: Albumentations & Imgaug libraries

