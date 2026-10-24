# Simple YOLO Comparison Project

## Setup
1. Install Python 3.9 or newer.
2. Open a terminal in this folder and run:
   pip install -r requirements.txt
3. Create a folder called `images` next to `yolo_project.py` and put your 10 images in it.

## Run
   python yolo_project.py

The first run downloads the 4 YOLO models (a few MB each).

## Output
- results/detections.json — all detections
- results/detections.csv — same data, opens in Excel
- results/plots/ — your images with boxes drawn, one per model and confidence score

## Things to try while learning
- Change `CONFIDENCES` to e.g. [0.1, 0.3, 0.5, 0.7, 0.9]
- Swap "n" for "s" in model names (yolov8s.pt) for bigger, more accurate models
- Open the CSV and count how many objects each model found
