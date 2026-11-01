"""FastAPI REST Service for EdgeInspect AI.

Provides high-throughput multi-domain industrial defect inspection and Llama advisory endpoints for
web applications, mobile apps, factory automation PLCs, and cloud microservices.
"""
import io
import sys
import uuid
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLASS_NAMES, DEFAULT_WEIGHTS, TARGETS  # noqa: E402
from domains import INDUSTRIAL_DOMAINS  # noqa: E402
from detect import EdgeInspector  # noqa: E402
from advisory import LlamaQualityAdvisor  # noqa: E402
from anomaly import IndustrialAnomalyDetector  # noqa: E402
from logger import QualityInspectionLogger  # noqa: E402
from hailo_adapter import HailoEdgeEngine  # noqa: E402

app = FastAPI(
    title="EdgeInspect AI — Multi-Domain Industrial Defect Inspection API",
    description=(
        "High-performance offline AI inspection engine for the Arm Bharat AI-SoC Challenge 2026–27. "
        "Supports PCB, Steel & Metal, Textile, Packaging, Automotive, and Pharma quality inspection."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for universal integration (web apps, React, Vue, Flutter, Postman)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_inspector: EdgeInspector = None
_advisor: LlamaQualityAdvisor = None
_anomaly_detector: IndustrialAnomalyDetector = None
_logger: QualityInspectionLogger = None
_hailo_engine: HailoEdgeEngine = None


def get_inspector(conf: float = 0.25) -> EdgeInspector:
    global _inspector
    if _inspector is None or _inspector.conf_threshold != conf:
        if not DEFAULT_WEIGHTS.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model checkpoint not found at {DEFAULT_WEIGHTS}. Train the model first."
            )
        _inspector = EdgeInspector(
            model_path=DEFAULT_WEIGHTS,
            conf_threshold=conf,
            device="cpu"
        )
    return _inspector


def get_advisor() -> LlamaQualityAdvisor:
    global _advisor
    if _advisor is None:
        _advisor = LlamaQualityAdvisor()
    return _advisor


def get_anomaly_detector(sensitivity: float = 0.50) -> IndustrialAnomalyDetector:
    global _anomaly_detector
    if _anomaly_detector is None or _anomaly_detector.sensitivity != sensitivity:
        _anomaly_detector = IndustrialAnomalyDetector(sensitivity=sensitivity)
    return _anomaly_detector


def get_logger() -> QualityInspectionLogger:
    global _logger
    if _logger is None:
        _logger = QualityInspectionLogger()
    return _logger


def get_hailo_engine() -> HailoEdgeEngine:
    global _hailo_engine
    if _hailo_engine is None:
        _hailo_engine = HailoEdgeEngine()
    return _hailo_engine


@app.get("/", tags=["General"])
def root_info() -> Dict[str, Any]:
    """Root metadata & service links."""
    return {
        "service": "EdgeInspect AI Multi-Domain Industrial Inspection API",
        "challenge": "Arm Bharat AI-SoC Challenge 2026–27 (PS5)",
        "version": "2.0.0",
        "documentation": "/docs",
        "supported_domains": list(INDUSTRIAL_DOMAINS.keys()),
        "endpoints": {
            "domains": "/api/v1/domains",
            "health": "/health",
            "inspect": "POST /api/v1/inspect",
            "logs": "/api/v1/logs",
            "hailo": "/api/v1/hailo"
        },
        "target_hardware": "Raspberry Pi 5 (Quad-Core Arm Cortex-A76 / Hailo-8 NPU)",
        "offline_ready": True
    }


@app.get("/health", tags=["General"])
def healthcheck() -> Dict[str, Any]:
    """Verify inference readiness, model availability, and edge targets."""
    weights_ready = DEFAULT_WEIGHTS.exists()
    return {
        "status": "healthy" if weights_ready else "degraded",
        "weights_present": weights_ready,
        "weights_path": str(DEFAULT_WEIGHTS),
        "target_spec": TARGETS,
        "supported_domains_count": len(INDUSTRIAL_DOMAINS),
        "supported_classes": CLASS_NAMES,
        "inference_device": "cpu",
        "reasoning_engine": "ExecuTorch (XNNPACK)"
    }


@app.get("/api/v1/domains", tags=["Industrial Domains"])
def list_domains() -> Dict[str, Any]:
    """List the 6 supported industrial application areas, standards, and defect types."""
    return {
        "count": len(INDUSTRIAL_DOMAINS),
        "domains": INDUSTRIAL_DOMAINS
    }


@app.get("/api/v1/classes", tags=["Inspection"])
def list_defect_classes() -> Dict[str, Any]:
    """List the 6 detectable PCB surface anomaly classes."""
    return {
        "count": len(CLASS_NAMES),
        "classes": [
            {"id": cid, "name": name}
            for cid, name in CLASS_NAMES.items()
        ]
    }


@app.post("/api/v1/inspect", tags=["Inspection"])
def inspect_part_image(
    file: UploadFile = File(..., description="Industrial part image file (JPEG or PNG)"),
    domain: str = Query("pcb", description="Industrial domain: pcb, steel_metal, textile_fabric, packaging, automotive, pharma"),
    mode: str = Query("detection", description="Inspection mode: 'detection' (YOLO) or 'anomaly' (OpenCV MVTec Heatmap)"),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Confidence / sensitivity threshold"),
    include_advisory: bool = Query(True, description="Run Llama ExecuTorch advisory engine"),
) -> Dict[str, Any]:
    """
    Execute full edge inspection on an uploaded industrial part image.
    Supports all 6 industrial domains with defect localization, thermal anomaly heatmapping,
    and Llama 3.2 ExecuTorch root cause & rework advisory.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "application/octet-stream"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type '{file.content_type}'. Please upload a JPG or PNG."
        )

    try:
        contents = file.file.read()
        pil_img = Image.open(io.BytesIO(contents))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image format: {exc}")

    temp_dir = PROJECT_ROOT / "reports" / "api_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    input_path = temp_dir / f"req_{job_id}.jpg"
    annotated_path = temp_dir / f"ann_{job_id}.jpg"
    heatmap_path = temp_dir / f"heat_{job_id}.jpg"

    try:
        pil_img.save(input_path, format="JPEG", quality=95)

        if mode == "anomaly":
            detector = get_anomaly_detector(sensitivity=conf)
            record = detector.detect_anomalies(
                input_path,
                output_heatmap_path=heatmap_path,
                output_annotated_path=annotated_path
            )
        else:
            inspector = get_inspector(conf=conf)
            record = inspector.inspect_image(input_path, output_annotated_path=annotated_path)

        response_payload = {
            "job_id": job_id,
            "filename": file.filename,
            "domain": domain,
            "mode": mode,
            "status": record["status"],
            "defects_found": record["defects_found"],
            "inference_ms": record["inference_ms"],
            "fps": round(1000.0 / max(1.0, record["inference_ms"]), 2),
            "detections": record.get("detections", []),
            "annotated_image_url": f"/api/v1/jobs/{job_id}/annotated",
        }

        if include_advisory:
            advisor = get_advisor()
            advisory_report = advisor.generate_advisory(record, domain=domain)
            response_payload["advisory"] = advisory_report

            # Log to production audit trail
            logger = get_logger()
            primary_d = "None"
            if record.get("detections"):
                first = record["detections"][0]
                primary_d = first.get("class_name", first.get("type", "Surface Anomaly"))

            logger.log_inspection(
                inspection_id=job_id,
                application_domain=domain,
                part_name=file.filename,
                status=record["status"],
                defects_count=record["defects_found"],
                inference_ms=record["inference_ms"],
                primary_defect=primary_d,
                severity=advisory_report.get("severity", "NORMAL"),
                disposition=advisory_report.get("disposition", "ACCEPT"),
                anomaly_score=record.get("anomaly_score", 0.0),
                detections=record.get("detections", [])
            )

        return response_payload

    finally:
        if input_path.exists():
            input_path.unlink()


@app.get("/api/v1/logs", tags=["Quality Logging"])
def get_inspection_logs(limit: int = 20) -> Dict[str, Any]:
    """Retrieve factory yield summary metrics and recent inspection audit records."""
    logger = get_logger()
    metrics = logger.get_summary_metrics()
    recent = logger.get_recent_records(limit=limit)
    return {
        "metrics": metrics,
        "recent_inspections": recent
    }


@app.get("/api/v1/accuracy", tags=["Quality Assurance"])
def get_accuracy_metrics() -> Dict[str, Any]:
    """Retrieve balanced 80-test-set ground-truth validation metrics and confusion breakdown."""
    acc_json = PROJECT_ROOT / "reports" / "accuracy_metrics.json"
    if acc_json.exists():
        with open(acc_json, "r", encoding="utf-8") as f:
            return json.load(f)
    from validate_accuracy import evaluate_balanced_dataset
    return evaluate_balanced_dataset()


@app.get("/api/v1/hardware", tags=["Edge Hardware"])
def get_hardware_profile(profile: Optional[str] = Query("realme_12x", description="Target profile: realme_12x, pi5_cpu, pi5_hailo, host_pc")) -> Dict[str, Any]:
    """Retrieve reconciled edge hardware platform specifications and runtime status."""
    from hardware import get_system_hardware_info
    return get_system_hardware_info(profile)


@app.get("/api/v1/hailo", tags=["Edge Hardware"])
def get_hailo_acceleration_profile() -> Dict[str, Any]:
    """Retrieve Hailo-8 / Hailo-8L NPU edge profile and benchmarks for Raspberry Pi 5."""
    engine = get_hailo_engine()
    return {
        "device": engine.get_device_info(),
        "benchmarks": engine.benchmark_pipeline()
    }


inspect_pcb_image = inspect_part_image


@app.get("/api/v1/jobs/{job_id}/annotated", tags=["Inspection"])
def get_job_annotated_image(job_id: str):
    """Retrieve the annotated visual result for a previously processed inspection job."""
    annotated_path = PROJECT_ROOT / "reports" / "api_temp" / f"ann_{job_id}.jpg"
    if not annotated_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotated image not found or expired.")
    return FileResponse(path=str(annotated_path), media_type="image/jpeg")
