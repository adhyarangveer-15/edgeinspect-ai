"""Configuration for EdgeInspect AI."""
from pathlib import Path

# If this file is in src/, root is parent.parent; if at root, root is parent
_current_dir = Path(__file__).resolve().parent
if _current_dir.name == "src":
    PROJECT_ROOT = _current_dir.parent
else:
    PROJECT_ROOT = _current_dir

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
DATA_YAML_PATH = DATA_DIR / "pcb_data.yaml"

# Keep these names synchronized with data/pcb_data.yaml and the trained model.
CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

IMAGE_SIZE = 640
DEVICE = "cpu"

TARGETS = {
    "precision_min": 0.90,
    "recall_min": 0.90,
    "map50_min": 0.85,
    "latency_max_ms": 300.0,
}

# Auto-locate model weights across nested or flat folder structures
candidate_weights = [
    MODELS_DIR / "baseline_nano" / "weights" / "best.pt",
    MODELS_DIR / "baseline_nano" / "weights" / "best.onnx",
    PROJECT_ROOT / "best.pt",
    PROJECT_ROOT / "best.onnx",
    PROJECT_ROOT / "models" / "best.pt",
    PROJECT_ROOT / "models" / "best.onnx",
]
DEFAULT_WEIGHTS = candidate_weights[0]
for p in candidate_weights:
    if p.exists():
        DEFAULT_WEIGHTS = p
        break
