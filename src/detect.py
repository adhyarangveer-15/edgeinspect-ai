"""Local PCB defect inference for EdgeInspect AI.

Important:
- Uses trained weights by default; it does NOT silently fall back to generic COCO weights.
- Current runtime is Ultralytics/PyTorch. This is not an ExecuTorch implementation.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import CLASS_NAMES, DEFAULT_WEIGHTS, IMAGE_SIZE

CLASS_COLORS = {
    0: (0, 140, 255),
    1: (255, 100, 0),
    2: (0, 0, 255),
    3: (0, 215, 255),
    4: (255, 0, 255),
    5: (180, 105, 255),
}


class EdgeInspector:
    def __init__(self, model_path=None, conf_threshold=0.25, device="cpu",
                 allow_pretrained_fallback=False):
        self.model_path = Path(model_path) if model_path else DEFAULT_WEIGHTS
        self.conf_threshold = float(conf_threshold)
        self.device = device

        if not self.model_path.exists():
            if allow_pretrained_fallback:
                self.model_path = Path("yolov8n.pt")
                print(
                    "WARNING: Using generic pretrained yolov8n.pt. "
                    "This is for pipeline smoke-testing only, not valid PCB defect results."
                )
            else:
                raise FileNotFoundError(
                    f"Trained weights not found: {self.model_path}\n"
                    "Train the PCB model or pass --weights to a verified checkpoint. "
                    "Generic yolov8n.pt is not a substitute for PCB-trained weights."
                )

        print(f"[EdgeInspector] Loading {self.model_path} on {device}...")
        self.model = YOLO(str(self.model_path))
        self._validate_class_names()

    def _validate_class_names(self):
        """Fail early if checkpoint class names conflict with the configured mapping."""
        model_names = getattr(self.model, "names", None)
        if not model_names:
            raise ValueError("Model checkpoint has no class-name metadata.")

        if isinstance(model_names, list):
            model_names = {i: name for i, name in enumerate(model_names)}

        configured_ids = set(CLASS_NAMES)
        model_ids = set(int(k) for k in model_names)
        if configured_ids != model_ids:
            raise ValueError(
                f"Class ID mismatch. Config IDs={sorted(configured_ids)}, "
                f"model IDs={sorted(model_ids)}. Check src/config.py and dataset YAML."
            )

        mismatches = {
            cid: (
                CLASS_NAMES[cid],
                str(model_names.get(cid, model_names.get(str(cid), ""))),
            )
            for cid in configured_ids
            if str(model_names.get(cid, model_names.get(str(cid), ""))).strip().lower()
            != CLASS_NAMES[cid].strip().lower()
        }
        if mismatches:
            details = "; ".join(
                f"id {cid}: config='{expected}', checkpoint='{actual}'"
                for cid, (expected, actual) in mismatches.items()
            )
            raise ValueError(
                "Checkpoint class names do not match src/config.py: " + details
            )

    def inspect_image(self, image_path, output_annotated_path=None):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"OpenCV could not decode image: {image_path}")
        h, w = img_bgr.shape[:2]

        # Warm-up once so first-run initialization does not distort the reported
        # steady-state model latency. Warm-up is kept separate from the metric.
        if not getattr(self, "_warmed_up", False):
            self.model.predict(
                source=img_bgr, conf=self.conf_threshold, imgsz=IMAGE_SIZE,
                device=self.device, verbose=False
            )
            self._warmed_up = True

        t0 = time.perf_counter()
        results = self.model.predict(
            source=img_bgr,
            conf=self.conf_threshold,
            imgsz=IMAGE_SIZE,
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_name = CLASS_NAMES.get(cls_id, str(result.names.get(cls_id, f"class_{cls_id}")))
            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
                "area_px": max(0, x2 - x1) * max(0, y2 - y1),
            })

            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            (_, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(text_h + 6, y1)
            cv2.rectangle(img_bgr, (x1, label_y - text_h - 6),
                          (x1 + 220, label_y), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, label_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        status = "DEFECTS_DETECTED" if detections else "NO_DEFECTS_DETECTED"
        banner_color = (0, 0, 200) if detections else (0, 130, 0)
        cv2.rectangle(img_bgr, (0, 0), (w, 35), banner_color, -1)
        banner = f"{status} | {inference_ms:.1f} ms | detections: {len(detections)}"
        cv2.putText(img_bgr, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2)

        if output_annotated_path:
            out_path = Path(output_annotated_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_path), img_bgr):
                raise IOError(f"Could not write annotated image: {out_path}")

        return {
            "image_name": image_path.name,
            "status": status,
            "defects_found": len(detections),
            "inference_ms": round(inference_ms, 2),
            "detections": detections,
            "model_path": str(self.model_path),
            "runtime": "Ultralytics/PyTorch",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a PCB image using trained weights")
    parser.add_argument("--image", required=True, help="Input PCB image")
    parser.add_argument("--weights", default=None, help="Path to trained model weights")
    parser.add_argument("--output", default=None, help="Path for annotated output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--allow-pretrained-fallback", action="store_true",
                        help="Allow generic yolov8n.pt for pipeline smoke-test only")
    args = parser.parse_args()

    inspector = EdgeInspector(
        model_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        allow_pretrained_fallback=args.allow_pretrained_fallback,
    )
    print(json.dumps(
        inspector.inspect_image(args.image, output_annotated_path=args.output),
        indent=2,
    ))
