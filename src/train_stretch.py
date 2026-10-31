"""
EdgeInspect AI: Stretch Target Fine-Tuning Script
Target: Precision: 85%+, Recall: 85%+, mAP@50: 80-85%+, F1-Score: 85%+
Dataset: 700 images (600 defect + 100 clean background negative samples)
Weights: Fine-tunes from models/improved_detector/weights/best.pt
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def train_stretch_detector():
    from ultralytics import YOLO

    config_path = PROJECT_ROOT / "data" / "balanced_train_config.yaml"
    initial_weights = PROJECT_ROOT / "models" / "improved_detector" / "weights" / "best.pt"

    print(f"Starting stretch fine-tuning on {config_path} using {initial_weights}...")
    t0 = time.perf_counter()

    model = YOLO(str(initial_weights))

    results = model.train(
        data=str(config_path),
        epochs=15,
        batch=16,
        imgsz=640,
        device="cpu",
        workers=0,
        project=str(PROJECT_ROOT / "models"),
        name="stretch_target_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        mosaic=0.0,
        scale=0.1,
        flipud=0.5,
        fliplr=0.5,
        plots=True,
        verbose=True
    )

    elapsed_min = (time.perf_counter() - t0) / 60.0
    print(f"\n[COMPLETE] Stretch target detector trained in {elapsed_min:.2f} minutes!")
    return results

if __name__ == "__main__":
    train_stretch_detector()
