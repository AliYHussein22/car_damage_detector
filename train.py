"""
train.py
--------
Fine-tune YOLOv8 on a car damage dataset from Roboflow.

Quick setup
-----------
1. Create a free account at https://roboflow.com
2. Search "car damage detection" at https://universe.roboflow.com
   Recommended: "Car-Damage-Detection" by Muhammad Farid (6 classes, ~2k images)
3. Click Export -> YOLOv8 -> "download zip to computer"
4. Unzip the downloaded file into data/car-damage/
5. Run: python train.py

Fine-tuned weights are saved to weights/car_damage.pt when training finishes.
Restart the Streamlit app afterwards and it will pick them up automatically.
"""

from ultralytics import YOLO
import os
import shutil

# training config — tweak these if you want to experiment
DATA_YAML   = "data/car-damage/data.yaml"   # path after unzipping the Roboflow export
BASE_MODEL  = "yolov8n.pt"                  # n = fastest; swap for yolov8s or yolov8m for better accuracy
EPOCHS      = 50
IMG_SIZE    = 640
BATCH       = 16
RUN_NAME    = "car_damage_v1"               # name for the runs/ output folder

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_YAML}.\n"
        "Download it from Roboflow Universe and unzip to data/car-damage/"
    )

model = YOLO(BASE_MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    name=RUN_NAME,
    patience=10,    # stop early if val loss stops improving for 10 epochs
    save=True,
    plots=True,     # saves confusion matrix, PR curve, etc. under runs/detect/
)

# copy the best checkpoint to the weights/ directory where the app expects it
best_checkpoint = f"runs/detect/{RUN_NAME}/weights/best.pt"
os.makedirs("weights", exist_ok=True)
shutil.copy(best_checkpoint, "weights/car_damage.pt")

print("\nDone! Fine-tuned weights saved to weights/car_damage.pt")
print("Restart the app and it will automatically use the new model.")
