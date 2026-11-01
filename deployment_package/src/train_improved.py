"""
EdgeInspect AI: Improved PCB Defect Detector Training Script
Target: Achieve >75% Precision, >75% Recall, >70% mAP@50, and >75% F1-score on balanced test set.
Dataset: Balanced 600-image training set (100 per class) + 90-image validation set.
Optimizations:
  - mosaic=0.0: Preserves small PCB defect geometry without 4x reduction.
  - scale=0.1: Minimal scale jitter for micro-defect preservation.
  - AdamW optimizer with cosine learning rate schedule.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def train_improved_detector(epochs: int = 15, batch: int = 16):
    from ultralytics import YOLO

    config_path = PROJECT_ROOT / "data" / "balanced_train_config.yaml"
    initial_weights = PROJECT_ROOT / "models" / "baseline_nano" / "weights" / "best.pt"
    if not initial_weights.exists():
        initial_weights = PROJECT_ROOT / "yolov8n.pt"

    print(f"Starting training on {config_path} using base weights {initial_weights}...")
    t0 = time.perf_counter()

    model = YOLO(str(initial_weights))

    results = model.train(
        data=str(config_path),
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device="cpu",
        workers=0,
        project=str(PROJECT_ROOT / "models"),
        name="improved_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        mosaic=0.0,
        scale=0.1,
        flipud=0.5,
        fliplr=0.5,
        plots=True,
        verbose=True
    )

    elapsed_min = (time.perf_counter() - t0) / 60.0
    print(f"\n[COMPLETE] Improved detector trained in {elapsed_min:.2f} minutes!")
    return results

if __name__ == "__main__":
    train_improved_detector(epochs=15, batch=16)
