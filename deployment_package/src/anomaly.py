"""
EdgeInspect AI: Industrial Surface Anomaly Detection Framework
Challenge: Arm Bharat AI-SoC Challenge 2026-27 (PS5: Industrial Defect Detection)
Target: Arm Cortex-A76 CPU / Edge Hardware via OpenCV & NumPy

Implements MVTec AD / KolektorSDD style unsupervised surface anomaly detection:
- High-frequency edge gradient & crack extraction (Sobel / Laplacian)
- Morphological Top-Hat / Black-Hat spatial anomaly filtering
- Adaptive background texture subtraction & local contrast deviation
- Color-coded thermal Anomaly Heatmap generation (Jet colormap)
- Automatic contour localization for unknown surface defects, cracks, and voids
"""
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np


class IndustrialAnomalyDetector:
    """
    Real-time edge visual anomaly detection engine.
    Detects cracks, surface scratches, porosity, and anomalies across:
    - Steel & Metal Surface (scratches, pits, rolled-in scale, crazing)
    - Textile & Fabric (holes, oil stains, broken yarn)
    - Packaging (seal breaches, punctures, dents)
    - Automotive Components (casting porosity, weld cracks, burrs)
    - Pharmaceutical Tablets & Capsules (chipped edges, capping, cracks)
    - Assembled & Bare PCBs (solder bridges, trace breaks, copper voids)
    """

    def __init__(self, sensitivity: float = 0.50, min_defect_area_px: int = 40):
        self.sensitivity = max(0.1, min(0.95, sensitivity))
        self.min_defect_area_px = min_defect_area_px

    def detect_anomalies(
        self,
        image_input,
        output_heatmap_path: Path = None,
        output_annotated_path: Path = None,
    ) -> Dict[str, Any]:
        """
        Processes an image and produces:
        - anomaly_score: [0.0 - 1.0] (overall anomaly index)
        - status: DEFECT_DETECTED or DEFECT_FREE
        - defects: List of localized anomaly bounding boxes and contours
        - heatmap_bgr: OpenCV Jet-colormap anomaly heatmap
        - annotated_bgr: Source image with bounding boxes & severity tags
        - latency_ms: Execution time on CPU
        """
        t0 = time.perf_counter()

        # Load or convert image to BGR numpy array
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Could not read image from {image_input}")
            img_name = Path(image_input).name
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
            img_name = "in_memory_image.jpg"
        else:
            raise TypeError("Expected image path or numpy array.")

        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (640, 640))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 1. Background illumination smoothing (Gaussian low-pass)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # 2. Multi-scale High-Frequency Texture & Crack Extraction
        # Gradient magnitude (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Morphological Top-Hat & Black-Hat (extracts small bright and dark surface anomalies)
        kernel_size = 9
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel)
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_kernel)
        morph_anom = cv2.add(top_hat, black_hat)

        # Local contrast deviation (high-pass difference from smooth background)
        diff_bg = cv2.absdiff(gray, blurred)
        diff_bg = cv2.equalizeHist(diff_bg)

        # 3. Fuse Anomaly Features into Unified Anomaly Energy Map
        fusion = 0.40 * grad_norm.astype(np.float32) + 0.35 * morph_anom.astype(np.float32) + 0.25 * diff_bg.astype(np.float32)
        fusion = cv2.normalize(fusion, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply spatial median filtering to reduce single-pixel sensor noise
        anomaly_map = cv2.medianBlur(fusion, 5)

        # Dynamic Thresholding based on user sensitivity
        # Sensitivity 0.50 -> threshold around 128
        threshold_val = int(255 * (1.0 - (self.sensitivity * 0.70)))
        _, thresh = cv2.threshold(anomaly_map, threshold_val, 255, cv2.THRESH_BINARY)

        # Find connected anomalous components / defect regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_x = w / 640.0
        scale_y = h / 640.0

        detections = []
        annotated = img_bgr.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_defect_area_px:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Map back to original image dimensions
                x1 = max(0, int(x * scale_x))
                y1 = max(0, int(y * scale_y))
                x2 = min(w, int((x + bw) * scale_x))
                y2 = min(h, int((y + bh) * scale_y))
                orig_area = int(area * scale_x * scale_y)

                # Local anomaly intensity score
                roi_anomaly = anomaly_map[y:y+bh, x:x+bw]
                mean_energy = float(np.mean(roi_anomaly)) / 255.0
                confidence = round(min(0.99, max(0.40, mean_energy * 1.25)), 4)

                defect_info = {
                    "anomaly_id": len(detections) + 1,
                    "bbox": [x1, y1, x2, y2],
                    "area_px": orig_area,
                    "confidence": confidence,
                    "energy_score": round(mean_energy, 3),
                    "type": "Surface Anomaly / Discontinuity"
                }
                detections.append(defect_info)

                # Draw bounding box and anomaly tag
                color = (0, 0, 255) if confidence > 0.7 else (0, 140, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ANOMALY #{len(detections)} ({confidence*100:.1f}%)"
                cv2.putText(
                    annotated, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
                )

        # Compute aggregate anomaly score for the entire part
        total_defective_px = int(np.count_nonzero(thresh))
        coverage_ratio = total_defective_px / (640.0 * 640.0)
        max_energy = float(np.max(anomaly_map)) / 255.0
        aggregate_anomaly_score = round(min(1.0, (coverage_ratio * 15.0) + (max_energy * 0.40)), 3)

        has_defect = len(detections) > 0
        status_str = "DEFECTS_DETECTED" if has_defect else "NO_DEFECTS_DETECTED"

        # 4. Generate Visual Thermal Heatmap (Jet Colormap)
        heatmap_resized = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        heatmap_full = cv2.resize(heatmap_resized, (w, h))

        # Blend Heatmap with original image (40% original, 60% heatmap)
        blended_heatmap = cv2.addWeighted(img_bgr, 0.45, heatmap_full, 0.55, 0)

        # Header Status Banner on annotated image
        banner_color = (0, 0, 180) if has_defect else (0, 140, 0)
        cv2.rectangle(annotated, (0, 0), (w, 32), banner_color, -1)
        banner_text = f"{status_str} | Anomaly Score: {aggregate_anomaly_score:.2f} | Anomalies: {len(detections)}"
        cv2.putText(annotated, banner_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Save files if paths provided
        if output_heatmap_path:
            Path(output_heatmap_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_heatmap_path), blended_heatmap)

        if output_annotated_path:
            Path(output_annotated_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_annotated_path), annotated)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "image_name": img_name,
            "status": status_str,
            "defects_found": len(detections),
            "anomaly_score": aggregate_anomaly_score,
            "detections": detections,
            "inference_ms": elapsed_ms,
            "fps": round(1000.0 / max(1.0, elapsed_ms), 1),
            "framework": "OpenCV Multi-Scale Texture Anomaly Engine (MVTec AD Profile)"
        }
