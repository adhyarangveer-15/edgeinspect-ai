"""FastAPI REST Service for EdgeInspect AI.

Provides high-throughput edge inspection and Llama advisory endpoints for
web applications, mobile apps, factory automation PLCs, and cloud microservices.
"""
import io
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLASS_NAMES, DEFAULT_WEIGHTS, TARGETS  # noqa: E402
from detect import EdgeInspector  # noqa: E402
from advisory import LlamaQualityAdvisor  # noqa: E402

app = FastAPI(
    title="EdgeInspect AI — Industrial PCB Defect Inspection API",
    description=(
        "High-performance offline AI inspection engine for the Arm Bharat AI-SoC Challenge 2026–27. "
        "Integrates lightweight vision defect detection with on-device ExecuTorch Llama 3.2 advisory."
    ),
    version="1.0.0",
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

# Cached singleton instances for maximum throughput
_inspector: EdgeInspector = None
_advisor: LlamaQualityAdvisor = None


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


@app.get("/", tags=["General"])
def root_info() -> Dict[str, Any]:
    """Root metadata & service links."""
    return {
        "service": "EdgeInspect AI Industrial Defect Detection API",
        "challenge": "Arm Bharat AI-SoC Challenge 2026–27 (PS5)",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "classes": "/api/v1/classes",
            "inspect": "POST /api/v1/inspect",
            "inspect_annotated": "POST /api/v1/inspect/annotated"
        },
        "target_hardware": "Raspberry Pi 5 (Quad-Core Arm Cortex-A76)",
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
        "supported_classes": CLASS_NAMES,
        "inference_device": "cpu",
        "reasoning_engine": "ExecuTorch (XNNPACK)"
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
def inspect_pcb_image(
    file: UploadFile = File(..., description="PCB image file (JPEG or PNG)"),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Confidence threshold"),
    include_advisory: bool = Query(True, description="Run Llama ExecuTorch advisory engine"),
) -> Dict[str, Any]:
    """
    Execute full edge inspection on an uploaded PCB image.
    
    Returns:
    - Defect localization (bounding boxes, class names, confidence scores)
    - Latency telemetry & FPS
    - Pass/Fail quality verdict
    - (Optional) Full Llama 3.2 IPC-A-610 Root Cause & Rework Advisory
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

    # Save to temporary buffer for inspector
    temp_dir = PROJECT_ROOT / "reports" / "api_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    input_path = temp_dir / f"req_{job_id}.jpg"
    annotated_path = temp_dir / f"ann_{job_id}.jpg"

    try:
        pil_img.save(input_path, format="JPEG", quality=95)
        inspector = get_inspector(conf=conf)
        vision_record = inspector.inspect_image(input_path, output_annotated_path=annotated_path)

        response_payload = {
            "job_id": job_id,
            "filename": file.filename,
            "status": vision_record["status"],
            "defects_found": vision_record["defects_found"],
            "inference_ms": vision_record["inference_ms"],
            "fps": round(1000.0 / max(1.0, vision_record["inference_ms"]), 2),
            "detections": vision_record["detections"],
            "annotated_image_url": f"/api/v1/jobs/{job_id}/annotated",
        }

        if include_advisory:
            advisor = get_advisor()
            advisory_report = advisor.generate_advisory(vision_record)
            response_payload["advisory"] = advisory_report

        return response_payload

    finally:
        # Clean up input temporary image
        if input_path.exists():
            input_path.unlink()


@app.post("/api/v1/inspect/annotated", tags=["Inspection"])
def inspect_and_return_image(
    file: UploadFile = File(..., description="PCB image file"),
    conf: float = Query(0.25, ge=0.05, le=0.95),
):
    """Inspect PCB image and stream back the visually annotated JPEG."""
    try:
        contents = file.file.read()
        pil_img = Image.open(io.BytesIO(contents))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {exc}")

    temp_dir = PROJECT_ROOT / "reports" / "api_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    input_path = temp_dir / f"req_{job_id}.jpg"
    annotated_path = temp_dir / f"ann_{job_id}.jpg"

    try:
        pil_img.save(input_path, format="JPEG", quality=95)
        inspector = get_inspector(conf=conf)
        inspector.inspect_image(input_path, output_annotated_path=annotated_path)

        return FileResponse(
            path=str(annotated_path),
            media_type="image/jpeg",
            filename=f"annotated_{file.filename}.jpg"
        )
    finally:
        if input_path.exists():
            input_path.unlink()


@app.get("/api/v1/jobs/{job_id}/annotated", tags=["Inspection"])
def get_job_annotated_image(job_id: str):
    """Retrieve the annotated visual result for a previously processed inspection job."""
    annotated_path = PROJECT_ROOT / "reports" / "api_temp" / f"ann_{job_id}.jpg"
    if not annotated_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotated image not found or expired.")
    return FileResponse(path=str(annotated_path), media_type="image/jpeg")
