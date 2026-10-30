# EdgeInspect AI: Real-Time PCB Defect Detection & Quality-Control Assistant

[![Challenge](https://img.shields.io/badge/Arm%20Bharat%20AI--SoC-PS5%20Challenge-blue.svg)](https://armbharatchallenge.com/)
[![Platform](https://img.shields.io/badge/Target-Raspberry%20Pi%205%20%7C%20Arm%20Cortex--A76-red.svg)]()
[![Model](https://img.shields.io/badge/Architecture-YOLOv8%20Nano%20%283.2M%20params%29-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

An offline, intelligent PCB inspection station powered by Arm-based edge hardware (Raspberry Pi 5), combining an edge-optimized computer vision defect localization engine with an on-device Llama advisory assistant executed via ExecuTorch.

Developed for the **Bharat AI-SoC Challenge 2026–27 (Problem Statement 5 — Industrial Defect Detection)** by Team VIT Vellore (Electronics Engineering / VLSI Design).

---

## 1. System Architecture

```text
[ High-Res PCB Image ]  (Controlled USB / Camera Capture)
         │
         ▼
[ Preprocessing Pipeline ]  (OpenCV: 640x640, Normalization, RGB alignment)
         │
         ▼
[ Edge Vision Engine ]  (YOLOv8 Nano, Arm Cortex-A76 CPU / NEON)
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
[ Structured Defect Output ]             [ Annotated PCB View ]
- Class (e.g., solder_bridge, spur)      - Bounding Boxes
- Confidence Score [0.0 - 1.0]           - Fault Category Labels
- Bounding Box [x1, y1, x2, y2]          - Real-Time Latency Banner
- Latency (ms) & Defect Status                     │
         │                                         │
         ▼                                         ▼
[ On-Device Llama Assistant ]             [ Operator Dashboard ]
(ExecuTorch on Arm CPU)                   (Offline Web UI / Streamlit)
- Root Cause Explanation                  - Visual Defect Inspection
- Severity Classification                 - CSV Inspection Log
- Recommended IPC-A-610 Rework Action     - Real-Time Edge Telemetry
```

---

## 2. Model Specification v1.0

### Objective
Develop a high-precision, low-latency offline computer vision model that localizes and classifies manufacturing defects on printed circuit boards using an Arm SoC (Raspberry Pi 5 CPU).

### Model Input Specification
| Parameter | Value / Initial Specification |
|---|---|
| **Input Format** | RGB Image |
| **Input Resolution** | $640 \times 640$ pixels |
| **Channels** | 3 (Red, Green, Blue) |
| **Input Tensor Shape** | `[1, 3, 640, 640]` |
| **Data Type** | `float32` / normalized $[0.0, 1.0]$ |
| **Inference Platform** | Raspberry Pi 5 CPU (Quad-core Arm Cortex-A76 @ 2.4 GHz) |
| **Cloud Dependency** | **None** (100% Offline Edge Inference) |

### Model Output Specification
| Field | Type | Description |
|---|---|---|
| `class_name` | string | Detected defect label (`mouse_bite`, `spur`, `missing_hole`, `short`, `open_circuit`, `spurious_copper`) |
| `confidence` | float | Prediction confidence score $[0.0, 1.0]$ |
| `bbox` | `[int, int, int, int]` | Bounding box coordinates $[x_1, y_1, x_2, y_2]$ in pixel space |
| `inference_ms` | float | Inference elapsed time on device |
| `status` | string | `DEFECT_DETECTED` or `DEFECT_FREE` |

### Engineering Targets vs. Baseline
| Metric | Engineering Goal | Baseline Model Target |
|---|---|---|
| **Precision** | $\ge 90.0\%$ | $\ge 85.0\%$ |
| **Recall** | $\ge 90.0\%$ | $\ge 85.0\%$ |
| **F1-Score** | $\ge 90.0\%$ | $\ge 85.0\%$ |
| **mAP@50** | $\ge 85.0\%$ | $\ge 80.0\%$ |
| **Median Inference Latency** | $\le 300\text{ ms}$ | $\le 200\text{ ms}$ (Nano) |
| **Model Size** | $\le 100\text{ MB}$ | $< 10\text{ MB}$ (YOLOv8n: $\approx 6.2\text{ MB}$) |
| **Peak RAM** | $\le 1024\text{ MB}$ | $< 350\text{ MB}$ |

---

## 3. Dataset Overview

The system uses the standard industrial PCB Defect Benchmark dataset (10,668 images):
- **Train Set:** 8,534 images (80%)
- **Validation Set:** 1,066 images (10%)
- **Held-out Test Set:** 1,068 images (10%)

### Defect Classes
1. `0: mouse_bite` — Edge erosion / notch defects
2. `1: spur` — Copper trace protrusion
3. `2: missing_hole` — Un-drilled via / mounting hole
4. `3: short` — Solder bridge / unintended bridge between traces
5. `4: open_circuit` — Broken / interrupted conductor trace
6. `5: spurious_copper` — Unetched copper residue

---

## 4. Repository Structure

```text
d:\arm bharat\
├── run_dashboard.bat         # [1-Click Launcher] Starts Dashboard & opens browser
├── run_api.bat               # [1-Click Launcher] Starts REST API on port 8000
├── share_public_link.bat     # [1-Click Worldwide Sharing] Live public HTTPS URL
├── DEPLOY_TO_WORLD.md        # Comprehensive worldwide accessibility guide
├── app.py                    # Root entrypoint for Hugging Face Spaces & Cloud
├── packages.txt              # Cloud Linux system dependencies (libgl1, libglib)
├── Dockerfile                # Production container specification
├── docker-compose.yml        # Multi-container orchestration (Dashboard + REST API)
├── data/
│   ├── pcb-defect-dataset/   # Dataset splits (train/val/test)
│   └── pcb_data.yaml         # YOLO dataset configuration
├── models/
│   └── baseline_nano/        # Model checkpoints (best.pt, best.onnx)
├── src/
│   ├── config.py             # Global constants & edge parameters
│   ├── advisory.py           # Llama 3.2 IPC-A-610 reasoning assistant (ExecuTorch)
│   ├── export_llama_pte.py   # ExecuTorch .pte exporter with XNNPACK
│   ├── dataset_summary.py    # Exploratory Data Analysis (EDA)
│   ├── train.py              # Model training pipeline
│   ├── evaluate.py           # Rigorous evaluation on held-out test split
│   ├── detect.py             # Inference & JSON / visual annotation engine
│   └── export.py             # ONNX / edge format converter
├── app/
│   ├── dashboard.py          # Operator inspection dashboard (Streamlit)
│   └── api.py                # Universal REST API & Swagger UI (FastAPI)
├── benchmarks/
│   ├── benchmark.py          # Latency, FPS, and RAM benchmarking suite
│   └── results.csv           # Measured hardware benchmarks
├── reports/
│   ├── dataset_eda_report.md # Statistical breakdown of classes
│   └── model_evaluation_test.md # Confusion matrix & PR metrics
├── tests/
│   ├── test_api.py           # REST API automated test suite
│   └── test_advisory.py      # ExecuTorch advisory unit tests
├── requirements.txt
└── README.md
```

---

## 5. How Anyone Can Use This AI Model (4 Easy Ways)

### 🌟 Way 1: For Everyone / Operators (1-Click Desktop Launcher)
**No coding or command-line experience needed!**
1. Open the project folder `D:\arm bharat` in Windows File Explorer.
2. Double-click **`run_dashboard.bat`**.
3. Your web browser will automatically open to `http://localhost:8501`.
4. Choose a sample test board or upload your own PCB photo (JPEG/PNG).
5. Click **"🚀 Run Inspection & Llama Advisory"** to view real-time defect localization and the IPC-A-610 repair advisory report!

*👉 **Want to share with someone right now on their mobile phone?** Double-click **`share_public_link.bat`** to get an instant worldwide HTTPS URL!*

---

### 🌐 Way 2: For Web, Mobile & IoT Developers (Production REST API)
Integrate this AI model directly into your mobile app, factory PLC, or web dashboard:
1. Double-click **`run_api.bat`** (or execute `python -m uvicorn app.api:app --reload`).
2. Open interactive Swagger API docs in your browser: **`http://localhost:8000/docs`**.
3. Available endpoints:
   - `GET /health`: Check model readiness and edge target specifications.
   - `GET /api/v1/classes`: Retrieve the 6 defect categories.
   - `POST /api/v1/inspect`: Upload a PCB photo to get defect bounding boxes, confidence, FPS, and the full Llama reasoning report in JSON.
   - `POST /api/v1/inspect/annotated`: Stream back a visually annotated JPEG with defect boxes and severity badges drawn.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/inspect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_pcb.jpg"
```

---

### 🐍 Way 3: For Python Developers & Data Scientists (Python Library)
Embed the model directly into any Python script or automation pipeline:
```python
from src.detect import EdgeInspector
from src.advisory import LlamaQualityAdvisor

# 1. Initialize offline edge vision detector
inspector = EdgeInspector(conf_threshold=0.25, device="cpu")

# 2. Inspect a PCB image
record = inspector.inspect_image("sample_board.jpg", output_annotated_path="annotated.jpg")
print(f"Status: {record['status']} | Defects: {record['defects_found']} | Latency: {record['inference_ms']:.1f}ms")

# 3. Generate Llama 3.2 ExecuTorch engineering advisory
advisor = LlamaQualityAdvisor()
advisory = advisor.generate_advisory(record)
print(f"IPC Verdict: {advisory['severity']}")
print(f"Disposition: {advisory['disposition']}")
for step in advisory['rework_instructions']:
    print(f" - {step}")
```

---

### 🐳 Way 4: For Cloud, DevOps & Worldwide 24/7 Hosting (Hugging Face & Docker)
Run seamlessly on any platform or host permanently 24/7 in the cloud:
```bash
# Build and run both Dashboard (port 8501) and REST API (port 8000)
docker-compose up --build
```
* **24/7 Free Worldwide Hosting:** Follow the step-by-step guide in **[`DEPLOY_TO_WORLD.md`](DEPLOY_TO_WORLD.md)** to deploy directly to **Hugging Face Spaces** or **Streamlit Community Cloud** with a permanent public web link!
* Uses the pre-configured root **`app.py`** and **`packages.txt`** for 1-click cloud builds.

---

## 6. Developer & Benchmark CLI Guide

### 1. Environment Setup
```bash
# Create and activate virtual environment
uv venv --python 3.11 .venv
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Dataset Exploratory Data Analysis (EDA)
```bash
python src/dataset_summary.py
```

### 3. Model Training
```bash
python src/train.py --epochs 10 --batch 16 --imgsz 640 --device cpu
```

### 4. Model Evaluation on Held-out Test Set
```bash
python src/evaluate.py --split test --device cpu
```

### 5. Single-Image Defect Inspection
```bash
python src/detect.py --image data/pcb-defect-dataset/test/images/sample.jpg --output reports/sample_annotated.jpg
```

### 6. Edge Benchmarking
```bash
python benchmarks/benchmark.py
```

### 7. Export to Edge Runtime (ONNX)
```bash
python src/export.py --format onnx
```

### 8. Llama Advisory Assistant via ExecuTorch
```bash
# Run standalone Llama reasoning pipeline on an inspection sample
python src/advisory.py

# Export Llama 3.2 model to ExecuTorch .pte format with XNNPACK
python src/export_llama_pte.py --model meta-llama/Llama-3.2-1B-Instruct --quant int8
```

### 9. Launch End-to-End Operator Dashboard
```bash
.venv\Scripts\streamlit run app/dashboard.py
```
