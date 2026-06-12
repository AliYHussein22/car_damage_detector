"""
src/detector.py
---------------
YOLOv8 wrapper — loads the model and runs inference on a single frame.

On first run this downloads YOLOv8n weights from Ultralytics (about 6 MB).
If you drop your own fine-tuned weights into weights/car_damage.pt the app
will automatically prefer those over the base model.

Custom weights source:
  https://universe.roboflow.com/car-damage-detection
  Model: car-damage-detection-fxkbd (YOLOv8, exported to .pt)
"""

from __future__ import annotations
import numpy as np
import cv2

# maps integer class ids to human-readable labels
# this mirrors the Roboflow car-damage dataset class order
LABEL_MAP = {
    0: "scratch",
    1: "dent",
    2: "crack",
    3: "broken-part",
    4: "flat-tyre",
    5: "shattered-glass",
}

# BGR colors for drawing bounding boxes — one per class
CLASS_COLORS = {
    "scratch":          (100, 200, 255),   # blue-ish
    "dent":             (255, 180,  60),   # orange
    "crack":            (100, 255, 180),   # green
    "broken-part":      (255,  80,  80),   # red
    "flat-tyre":        (220, 100, 255),   # purple
    "shattered-glass":  (255, 255, 100),   # yellow
}


def load_model():
    """
    Load a YOLOv8 model.

    Priority order:
      1. weights/car_damage.pt  — your fine-tuned weights (placed here after training)
      2. yolov8n.pt             — base model, auto-downloaded on first run

    To swap in a custom model just drop the .pt file in the weights/ directory
    and restart the app — no code changes needed.
    """
    from ultralytics import YOLO
    import os

    custom_path = os.path.join(os.path.dirname(__file__), "..", "weights", "car_damage.pt")
    if os.path.exists(custom_path):
        model = YOLO(custom_path)
        print(f"Loaded fine-tuned model: {custom_path}")
    else:
        model = YOLO("yolov8n.pt")
        print("Using base YOLOv8n — add fine-tuned weights to weights/car_damage.pt for better results")

    return model


def run_detection(model, img_rgb: np.ndarray, confidence: float = 0.25) -> tuple[np.ndarray, list]:
    """
    Run YOLOv8 on a single RGB image.

    Parameters
    ----------
    model      : loaded YOLO model (from load_model)
    img_rgb    : H x W x 3 numpy array in RGB order
    confidence : minimum confidence threshold, 0–1

    Returns
    -------
    annotated  : RGB image with colored bounding boxes drawn on it
    detections : list of dicts — each has 'label', 'confidence', and 'box' (x1,y1,x2,y2)
    """
    results    = model(img_rgb, conf=confidence, verbose=False)
    annotated  = img_rgb.copy()
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls  = int(box.cls[0])

            # prefer the model's own class names if available (handles custom models cleanly)
            if hasattr(result, "names") and cls in result.names:
                label = result.names[cls].lower().replace(" ", "-")
            else:
                label = LABEL_MAP.get(cls, f"damage-{cls}")

            color = CLASS_COLORS.get(label, (200, 200, 200))

            # draw the bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # draw a filled label background so the text is readable over any background
            text = f"{label}  {conf:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            detections.append({
                "label":      label,
                "confidence": conf,
                "box":        (x1, y1, x2, y2),
            })

    return annotated, detections
