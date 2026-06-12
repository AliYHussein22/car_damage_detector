# CarDamage AI

A YOLOv8-powered car damage detection app built with Streamlit. Upload a photo or video of a car and get an instant damage report with bounding boxes, damage classification, and an overall severity score.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

- Detects car damage in photos and videos using YOLOv8
- Draws colored bounding boxes around each detected damage area
- Classifies damage into 6 types: scratch, dent, crack, broken part, flat tyre, shattered glass
- Computes an overall severity score (0–10) and labels it Low, Medium, or High
- Supports adjustable confidence threshold via a sidebar slider

---

## Demo

| Input | Detection | Severity |
|---|---|---|
| Car with scratch and dent | Colored bounding boxes drawn on panel | Low — 3.1 / 10 |
| Car with shattered glass and broken bumper | Multiple boxes detected | High — 8.4 / 10 |

---

## Quick start

**1. Clone the repo**
```bash
git clone https://github.com/AliYHussein22/car_damage_detector.git
cd car_damage_detector
```

**2. Create a virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

On first run, YOLOv8 downloads base weights (~6 MB) automatically. For real car damage detection, fine-tune the model first — see the section below.

---

## Fine-tuning on car damage data (recommended)

The app works out of the box with base YOLOv8 weights, but it won't detect car-specific damage well without fine-tuning. The base model was trained on COCO (everyday objects), not car damage.

**1. Get the dataset**

Go to [Roboflow Universe](https://universe.roboflow.com) and search for "car damage detection". A good starting point is the dataset by **btp** (4k images, 6 classes matching this project).

Download it in YOLOv8 format and unzip to:
```
data/car-damage/
    train/
    valid/
    test/
    data.yaml
```

**2. Train**
```bash
python train.py
```

Training takes ~15–30 min on a GPU or ~2 hours on CPU. Fine-tuned weights are saved to `weights/car_damage.pt` automatically when done.

**3. Restart the app** — it auto-detects the fine-tuned weights on startup and switches to them.

---

## Project structure

```
car_damage_detector/
|-- app.py              # Streamlit UI — handles file upload, layout, and report rendering
|-- train.py            # Fine-tuning script — downloads base model and trains on your dataset
|-- requirements.txt    # Python dependencies
|-- src/
|   |-- detector.py     # YOLOv8 wrapper — loads model, runs inference, draws bounding boxes
|   `-- severity.py     # Scoring logic — converts detections into a 0-10 severity score
|-- data/               # Put your Roboflow dataset here (not tracked by git)
`-- weights/            # Fine-tuned weights go here (not tracked by git)
```

---

## Damage classes

| Class | Bounding box color | Severity weight |
|---|---|---|
| scratch | Blue | 1.0 |
| dent | Orange | 1.5 |
| crack | Green | 2.0 |
| flat-tyre | Purple | 2.0 |
| broken-part | Red | 3.0 |
| shattered-glass | Yellow | 2.5 |

Severity score = weighted sum of detection confidences, normalised to 0–10.

---

## Robustness notes

| Condition | Performance |
|---|---|
| Multi-scale zoom | Good — YOLOv8 trains at multiple scales by default |
| Mild rotation | Decent — works for typical car photo angles |
| Heavy rotation (45°+) | Drops noticeably |
| Mild blur | Fine |
| Heavy motion blur | Degrades confidence, especially on scratches |

Best results: shoot straight-on from 1–3 meters, decent lighting, steady hand.

---

## Requirements

```
ultralytics>=8.0
streamlit>=1.35
opencv-python>=4.9
Pillow>=10.0
numpy>=1.26
```

---

## License

MIT — do whatever you want with it, just don't blame me if the model misses a dent on your car.
