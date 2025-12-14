  ![YOLO Output 1](/Sample.png)
  Weather-Robust Vehicle Detection on BDD100K (YOLOv11)
  This project trains and evaluates a YOLOv11n-based vehicle detector (car, truck, bus) on a 10% subset of the BDD100K dataset, with a clear-weather baseline and mixed-weather testing using test-time augmentation (TTA).

  Clear-weather val mAP50: 0.458

  Mixed-weather test mAP50 (baseline): 0.466

  Mixed-weather test mAP50 (+TTA): 0.499

1. Installation & Environment Setup
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install ultralytics jupyter notebook matplotlib numpy tqdm pyyaml
```
  Ultralytics YOLO docs for reference: https://docs.ultralytics.com

2. Dataset Setup (BDD100K 10% Subset)
   Download BDD100K images and labels from the official website and place the 10% subset in Google Drive (or local disk) in this structure:
```
<drive_or_local_root>/
  Embitel/
    subset_10/
      images/
        train/*.jpg
        val/*.jpg
        test/*.jpg
      labels/
        train/*.json
        val/*.json
        test/*.json
```
3. How to Load BDD100K & Run Preprocessing (Notebooks)
   All steps are organized as Jupyter/Colab notebooks:

notebooks/01_eda_and_preprocessing.ipynb

  EDA: class distribution, box sizes, weather/time-of-day, object density.

  Data preprocessing:

  Filter clear-weather images from subset_10/labels/train (using attributes["weather"] == "clear").

  Keep only car, truck, bus objects.

  Convert BDD100K box2d → YOLO txt format (class cx cy w h normalized).

  Create clear-weather YOLO dataset:
```
Embitel/
  yolo_dataset/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    dataset.yaml
```
  dataset.yaml example:
```
path: /content/drive/MyDrive/Embitel/yolo_dataset
train: images/train
val: images/val

nc: 3
names: ['car', 'truck', 'bus']
```
  notebooks/02_build_mixed_test_set.ipynb

  Constructs the mixed-weather test set from subset_10/test:
```
Embitel/
  test_mixed_v2/
    images/*.jpg
    labels/*.txt    # YOLO-format labels from original JSONs
    test.yaml
```
  test.yaml example:
```
path: /content/drive/MyDrive/Embitel/test_mixed_v2
train: images   # dummy, used only to satisfy YOLO format
val: images

nc: 3
names: ['car', 'truck', 'bus']
```
4. Training: Clear-Weather Baseline
   Use notebooks/03_train_yolov11_clear_weather.ipynb to train the baseline model.

   Core code:
```
from ultralytics import YOLO

# Load COCO-pretrained YOLOv11n
model = YOLO("yolo11n.pt")

# Train on clear-weather, vehicle-only dataset
results = model.train(
    data="/content/drive/MyDrive/Embitel/yolo_dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,                 # GPU in Colab
    name="yolo11n_clear_vehicles",
    patience=10,
    save=True,
    plots=True
)

# Best weights
best_path = "/content/runs/detect/yolo11n_clear_vehicles/weights/best.pt"
# Recommended: copy to Drive
!cp -r /content/runs/detect/yolo11n_clear_vehicles /content/drive/MyDrive/Embitel/
```
   This trains a clear-weather baseline detector on 2922 train + 731 val images (weather == clear, classes: car/truck/bus).

5. Evaluation: Mixed-Weather Test + TTA
    Use notebooks/04_eval_and_tta.ipynb to evaluate on the mixed-weather test set.

5.1. Baseline Evaluation (Mixed Weather)
```
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/Embitel/best.pt")  # or best.pt path from training

mixed_test_yaml = "/content/drive/MyDrive/Embitel/test_mixed_v2/test.yaml"

baseline = model.val(
    data=mixed_test_yaml,
    imgsz=640,
    plots=True,
    save_json=True,
    name="test_baseline"
)

print(f"Baseline mAP50 (mixed): {baseline.box.map50:.3f}")
print(f"Baseline mAP50-95:      {baseline.box.map:.3f}")
print(f"Precision:              {baseline.box.mp:.3f}")
print(f"Recall:                 {baseline.box.mr:.3f}")
```
5.2. Test-Time Augmentation (TTA) Evaluation
```
tta = model.val(
    data=mixed_test_yaml,
    imgsz=800,          # single higher resolution; YOLO warns but accepts int
    augment=True,       # enables TTA (flips/scale/brightness)
    half=True,          # FP16 for speed (if supported)
    plots=True,
    name="test_tta"
)

print(f"TTA mAP50 (mixed): {tta.box.map50:.3f}")
print(f"TTA mAP50-95:      {tta.box.map:.3f}")
```
6. How to Interpret the Metrics
  mAP50: Mean Average Precision at IoU 0.5 (standard “accuracy” for object detection).

  mAP50–95: COCO-style mAP averaged over IoU thresholds [0.5, 0.95]; stricter, measures localization quality.

  Precision (P): Of all predicted boxes, how many are correct.

  Recall (R): Of all ground-truth boxes, how many are detected.

  In this project:

  Clear-weather training and mixed-weather testing show no performance drop (0.458 → 0.466 mAP50), indicating strong weather robustness.

  TTA further boosts mixed-weather performance to 0.499 mAP50 without retraining, improving small-object and adverse-condition robustness.
