"""
EdgeInspect AI: Hailo Dataflow Compiler (DFC) Script
Compiles YOLOv8 ONNX defect detection model into a Hailo Executable Format (.hef)
for the Raspberry Pi 5 AI Kit (Hailo-8 / Hailo-8L).
"""
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = PROJECT_ROOT / "models" / "baseline_nano" / "weights" / "best.onnx"
DEFAULT_HEF = PROJECT_ROOT / "models" / "baseline_nano" / "weights" / "best.hef"


def generate_hailo_compilation_workflow(onnx_path=DEFAULT_ONNX, output_hef=DEFAULT_HEF, target="hailo8l"):
    """
    Generates the exact Hailo Dataflow Compiler commands and python workflow.
    Target: 'hailo8l' (Raspberry Pi AI Kit) or 'hailo8' (Standard Hailo-8 M.2).
    """
    script_content = f"""#!/usr/bin/env bash
# Hailo Dataflow Compiler Pipeline for EdgeInspect AI
# Target: Raspberry Pi 5 AI Kit ({target.upper()})

set -e

ONNX_MODEL="{onnx_path}"
OUTPUT_HEF="{output_hef}"
ALLS_SCRIPT="models/yolov8_hailo.alls"

echo "=========================================================="
echo " EdgeInspect AI: Hailo-8 / Hailo-8L Compilation Pipeline"
echo "=========================================================="

# 1. Parse ONNX Model into Hailo HAR (Hailo Archive)
echo "[Step 1/3] Parsing ONNX model with Hailo Parser..."
hailo parser onnx "$ONNX_MODEL" \\
    --hw-arch {target} \\
    --har-path models/yolov8_parsed.har \\
    --model-name yolov8n_pcb

# 2. INT8 Quantization & Calibration
echo "[Step 2/3] Quantizing model with calibration dataset..."
hailo optimize models/yolov8_parsed.har \\
    --hw-arch {target} \\
    --har-path models/yolov8_quantized.har

# 3. Compile to Hailo Executable Format (.hef)
echo "[Step 3/3] Compiling optimized HAR to HEF binary..."
hailo compiler models/yolov8_quantized.har \\
    --hw-arch {target} \\
    --output-path "$OUTPUT_HEF"

echo "=========================================================="
echo " Compilation SUCCESSFUL! Target binary: $OUTPUT_HEF"
echo " Deploy directly to Raspberry Pi 5 AI Kit with pyhailort!"
echo "=========================================================="
"""
    return script_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hailo-8 HEF Compilation Workflow Generator")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="Path to best.onnx")
    parser.add_argument("--output", default=str(DEFAULT_HEF), help="Output .hef file")
    parser.add_argument("--target", default="hailo8l", choices=["hailo8l", "hailo8"], help="Hailo hardware target")
    args = parser.parse_args()

    script = generate_hailo_compilation_workflow(args.onnx, args.output, args.target)
    out_script_path = PROJECT_ROOT / "scripts" / "compile_hailo.sh"
    out_script_path.parent.mkdir(parents=True, exist_ok=True)
    out_script_path.write_text(script, encoding="utf-8")
    print(f"Hailo compilation script written to {out_script_path}")
