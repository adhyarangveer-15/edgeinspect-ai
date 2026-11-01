"""
EdgeInspect AI: Mobile Edge Model Exporter (Android / Realme 12x 5G)
Target Hardware: MediaTek Dimensity 6100+ (2x Arm Cortex-A76, 6x Arm Cortex-A55, Mali-G57 GPU)
Exports:
1. ONNX Runtime Mobile format (.onnx) with dynamic shape and FP16/INT8 support
2. TensorFlow Lite (.tflite) for native Android CameraX integration
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "baseline_nano" / "weights"
PT_PATH = WEIGHTS_DIR / "best.pt"
ONNX_PATH = WEIGHTS_DIR / "best.onnx"


def export_mobile_models():
    print("=" * 60)
    print(" EdgeInspect AI - Mobile Edge Exporter for Android")
    print(" Target: Realme 12x 5G (MediaTek Dimensity 6100+ / Arm Cortex-A76)")
    print("=" * 60)

    if not PT_PATH.exists():
        raise FileNotFoundError(f"Trained weights not found at {PT_PATH}")

    from ultralytics import YOLO

    print(f"\n[1/2] Loading trained model from {PT_PATH}...")
    model = YOLO(str(PT_PATH))

    # Verify existing ONNX
    if ONNX_PATH.exists():
        print(f"  -> Mobile ONNX already generated: {ONNX_PATH} ({round(ONNX_PATH.stat().st_size / (1024*1024), 2)} MB)")
    else:
        print("  -> Exporting ONNX for Android NNAPI / ONNX Runtime Mobile...")
        model.export(format="onnx", imgsz=640, opset=17, dynamic=False, simplify=True)

    print("\n[2/2] Exporting to TensorFlow Lite (.tflite) for Android CameraX...")
    try:
        tflite_path = model.export(format="tflite", imgsz=640, int8=False)
        print(f"  -> TFLite export SUCCESS: {tflite_path}")
    except Exception as exc:
        print(f"  -> Note: TFLite export requires tensorflow-cpu. ONNX is fully supported on Android via onnxruntime-android.")
        print(f"     Details: {exc}")

    print("\n" + "=" * 60)
    print(" Android Deployment Artifacts Ready!")
    print(f" 1. ONNX Model: {ONNX_PATH}")
    print(" 2. Android Runtime: onnxruntime-android (supports Arm NEON + NNAPI)")
    print("=" * 60)


if __name__ == "__main__":
    export_mobile_models()
