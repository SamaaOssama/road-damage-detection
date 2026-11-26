# Road Damage Object Detection — YOLO Comparison

Train, compare, tune and deploy YOLO models that detect road damage
(alligator, block, crack, longitudinal, pothole, transverse).

**Best model:** YOLOv10s + AdamW — test mAP50 0.606, mAP50-95 0.372, 63.6 FPS on a T4 GPU, 16.5 MB.

## Project structure

```
road-damage-detection/
├── README.md
├── notebook/
│   └── road_damage_yolo.ipynb      # full pipeline: data → training → tuning → evaluation → discussion
├── results/
│   ├── comparison.csv              # all metrics for the 5 models (mAP, P, R, F1, IoU, FPS, size, GFLOPs)
│   ├── per_class.csv               # precision / recall / AP per class per model
│   ├── per_class_AP50.csv          # AP50 table: classes × models
│   ├── epochs_30_vs_60.csv         # effect of training length
│   ├── class_distribution.csv      # boxes per class per split
│   ├── train_time_minutes.json
│   ├── best_model.onnx             # ONNX export of the best model
│   └── figures/                    # all charts, confusion matrix, sample predictions
└── app/
    ├── app.py                      # Gradio deployment app
    ├── requirements.txt
    ├── best_model.pt               # trained weights (YOLOv10s + AdamW)
    ├── best_model.txt              # name of the best model
    └── examples/                   # (optional) put a few test images here for demo examples
```

## Dataset
Roboflow Universe "road-damage" (CC BY 4.0), forked and versioned:
6,757 images, 4,930 train / 1,146 valid / 681 test, resized to 640×640.
The `edge` class (22 boxes, none in test) was dropped → 6 classes.

## Models compared
| Run | Description |
|---|---|
| yolov8s, yolo11s, yolov10s | baselines, identical settings (60 epochs, SGD lr0 0.01, imgsz 640, seed 0) |
| yolov10s_tune_adamw | AdamW, lr0 0.001 |
| yolov10s_tune_cos_aug | SGD + cosine LR, mixup 0.1, scale 0.7, close_mosaic 5 |

## Run the notebook
Kaggle, GPU T4 x2, Internet on, secret `ROBOFLOW_API_KEY`. Set `WORKSPACE` / `PROJECT` in the config cell, then *Save Version → Save & Run All*.

## Run the app
```bash
cd app
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:7860 — upload an image, use the webcam, or process a video.
