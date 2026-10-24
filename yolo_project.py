"""
Simple YOLO comparison project (beginner version)
=================================================
What it does:
  1. Reads your images from the "images" folder (put 10 images there)
  2. Loads 4 YOLO versions (v5, v8, v10, v11)
  3. Runs object detection on every image with every model
  4. Repeats it for several confidence scores (0.25, 0.50, 0.75)
  5. Saves all detections to results/detections.json and results/detections.csv
  6. Draws the boxes on the images and saves them in results/plots/

How to run:
  pip install -r requirements.txt
  python yolo_project.py
"""

import os          # for working with folders and file paths
import json        # for saving JSON files
import csv         # for saving CSV files
import cv2         # OpenCV: reads images and draws boxes
from ultralytics import YOLO   # the library that runs YOLO models

# ---------------- SETTINGS (change these to experiment) ----------------
IMAGE_FOLDER = "images"         # where your 10 images are
OUTPUT_FOLDER = "results"       # where results will be saved

# 4 YOLO versions. "n" = nano (smallest and fastest).
# The files download automatically the first time.
MODELS = {
    "YOLOv5":  "yolov5nu.pt",
    "YOLOv8":  "yolov8n.pt",
    "YOLOv10": "yolov10n.pt",
    "YOLOv11": "yolo11n.pt",
}

# Confidence scores to try. A detection is kept only if the model is
# at least this sure. Higher = fewer but more reliable boxes.
CONFIDENCES = [0.25, 0.50, 0.75]
# -----------------------------------------------------------------------


def get_image_paths(folder):
    """Return a list of all image files inside the folder."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(extensions):
            files.append(os.path.join(folder, name))
    return files


def draw_boxes(image, detections):
    """Draw a rectangle and a label for each detection on the image."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        # green rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # label text above the box
        cv2.putText(image, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def main():
    # Make output folders if they don't exist
    plots_folder = os.path.join(OUTPUT_FOLDER, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    images = get_image_paths(IMAGE_FOLDER)
    print(f"Found {len(images)} images")

    all_results = []   # every single detection will be stored here

    # Loop 1: every model
    for model_name, model_file in MODELS.items():
        print(f"\nLoading {model_name} ...")
        model = YOLO(model_file)

        # Loop 2: every confidence score
        for conf in CONFIDENCES:

            # Loop 3: every image
            for image_path in images:
                image_name = os.path.basename(image_path)

                # Run detection. verbose=False keeps the console clean.
                result = model.predict(image_path, conf=conf, verbose=False)[0]

                # Turn YOLO output into simple Python dictionaries
                detections = []
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    detection = {
                        "model": model_name,
                        "confidence_threshold": conf,
                        "image": image_name,
                        "class_name": model.names[class_id],
                        "confidence": round(float(box.conf[0]), 3),
                        "box": [int(v) for v in box.xyxy[0].tolist()],  # x1,y1,x2,y2
                    }
                    detections.append(detection)

                all_results.extend(detections)
                print(f"{model_name} | conf={conf} | {image_name}: "
                      f"{len(detections)} objects")

                # Draw boxes and save the picture
                img = cv2.imread(image_path)
                img = draw_boxes(img, detections)
                out_name = f"{model_name}_conf{conf}_{image_name}"
                cv2.imwrite(os.path.join(plots_folder, out_name), img)

    # ---------- Save JSON ----------
    json_path = os.path.join(OUTPUT_FOLDER, "detections.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---------- Save CSV ----------
    csv_path = os.path.join(OUTPUT_FOLDER, "detections.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "confidence_threshold", "image",
                         "class_name", "confidence", "x1", "y1", "x2", "y2"])
        for d in all_results:
            writer.writerow([d["model"], d["confidence_threshold"], d["image"],
                             d["class_name"], d["confidence"], *d["box"]])

    print(f"\nDone! {len(all_results)} detections saved.")
    print(f"JSON : {json_path}")
    print(f"CSV  : {csv_path}")
    print(f"Images with boxes: {plots_folder}")


# This line means: run main() only when this file is run directly
if __name__ == "__main__":
    main()
