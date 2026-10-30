"""Streamlit dashboard for EdgeInspect AI with ExecuTorch & Llama Assistant."""
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

_this_file = Path(__file__).resolve()
if _this_file.parent.name == "app":
    PROJECT_ROOT = _this_file.parent.parent
else:
    PROJECT_ROOT = _this_file.parent

SRC_DIR = PROJECT_ROOT / "src"
for p in [PROJECT_ROOT, SRC_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import DEFAULT_WEIGHTS, DATA_DIR, TARGETS  # noqa: E402
from advisory import LlamaQualityAdvisor  # noqa: E402

st.set_page_config(
    page_title="EdgeInspect AI — PCB Defect Inspection & Llama Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 EdgeInspect AI — PCB Defect Inspection Station")
st.caption("Arm Bharat AI-SoC Challenge 2026–27 | PS5: Industrial Defect Detection | Raspberry Pi 5 (Arm SoC)")

# Sidebar Configuration
st.sidebar.header("⚙️ Inspection Settings")
conf_threshold = st.sidebar.slider(
    "Detection confidence threshold", min_value=0.10, max_value=0.90,
    value=0.25, step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Edge Deployment Profile")
st.sidebar.markdown("""
- **Vision Runtime:** YOLOv8 Nano (Arm CPU / ONNX)
- **Reasoning Engine:** **ExecuTorch** (XNNPACK Backend)
- **Advisory Model:** **Llama 3.2** IPC-A-610 Quality Assistant
- **Target Hardware:** Raspberry Pi 5 (Arm Cortex-A76)
- **Offline Assurance:** **100% On-Device Execution**
""")
st.sidebar.info(f"Target Vision Latency: ≤ {TARGETS['latency_max_ms']:.0f} ms/image")

sample_dir = DATA_DIR / "pcb-defect-dataset" / "test" / "images"
sample_images = []
try:
    if sample_dir.exists():
        sample_images = sorted(list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")))
except Exception:
    sample_images = []

if not sample_images:
    bundled_dir = PROJECT_ROOT / "app" / "sample_images"
    if bundled_dir.exists():
        sample_images = sorted(list(bundled_dir.glob("*.jpg")) + list(bundled_dir.glob("*.png")))

source = st.radio(
    "Image source",
    ["Choose test-set image", "Upload PCB image"],
    horizontal=True,
)

selected_path = None
uploaded_file = None
if source == "Upload PCB image":
    uploaded_file = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])
else:
    if sample_images:
        selected_path = st.selectbox("Select test-set image", sample_images, format_func=lambda p: p.name)
    else:
        st.warning("No sample images found. Please upload a PCB image.")

image = None
image_name = "uploaded_pcb.jpg"
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name
    except Exception as exc:
        st.error(f"Could not open uploaded image: {exc}")
elif selected_path is not None:
    try:
        image = Image.open(selected_path).convert("RGB")
        image_name = selected_path.name
    except Exception as exc:
        st.error(f"Could not open selected image: {exc}")

if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Input PCB Image")
        st.image(image, use_container_width=True, caption=f"Source: {image_name}")

    run = st.button("🚀 Run Inspection & Llama Advisory", type="primary")
    if run:
        if not DEFAULT_WEIGHTS.exists():
            st.error(
                f"Trained weights were not found at `{DEFAULT_WEIGHTS}`. "
                "Train the PCB model or place a verified trained checkpoint there."
            )
        else:
            try:
                from detect import EdgeInspector

                with st.spinner("Step 1/2: Running vision inference on Arm CPU..."):
                    inspector = EdgeInspector(
                        model_path=DEFAULT_WEIGHTS,
                        conf_threshold=conf_threshold,
                        device="cpu",
                    )
                    reports_dir = PROJECT_ROOT / "reports"
                    reports_dir.mkdir(parents=True, exist_ok=True)
                    unique_id = uuid.uuid4().hex[:10]
                    input_path = reports_dir / f"input_{unique_id}.jpg"
                    output_path = reports_dir / f"annotated_{unique_id}.jpg"
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(input_path, format="JPEG", quality=95)
                    record = inspector.inspect_image(input_path, output_annotated_path=output_path)

                with col2:
                    st.subheader("🎯 Vision Defect Localization")
                    st.image(str(output_path), use_container_width=True, caption=f"Status: {record['status']}")

                # Telemetry Bar
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                has_defect = record["defects_found"] > 0
                m1.metric("Detection Verdict", "DEFECT DETECTED" if has_defect else "PASS / CLEAR",
                          delta="FAULTS FOUND" if has_defect else "NORMAL", delta_color="inverse")
                m2.metric("Defects Located", f"{record['defects_found']} anomalies")
                m3.metric("Vision Latency", f"{record['inference_ms']:.1f} ms",
                          delta=f"{TARGETS['latency_max_ms'] - record['inference_ms']:.1f} ms headroom")
                m4.metric("Throughput", f"{1000.0 / max(1.0, record['inference_ms']):.1f} FPS")

                # Step 2: Llama ExecuTorch Advisory Engine
                with st.spinner("Step 2/2: Executing on-device Llama reasoning via ExecuTorch..."):
                    advisor = LlamaQualityAdvisor()
                    advisory_result = advisor.generate_advisory(record)

                st.markdown("---")
                st.subheader("🤖 Llama 3.2 ExecuTorch Reasoning Engine")
                st.caption(f"Reasoning Runtime: **{advisory_result['engine']}** | Advisory Latency: **{advisory_result['reasoning_latency_ms']:.2f} ms**")

                tab_report, tab_rework, tab_prompt = st.tabs([
                    "📋 Formal Engineering Report",
                    "🛠️ Operator Rework Checklist",
                    "🔍 Llama Prompt & ExecuTorch Telemetry"
                ])

                with tab_report:
                    st.markdown(advisory_result["report_markdown"])
                    st.download_button(
                        "📥 Download Technical Report (.md)",
                        data=advisory_result["report_markdown"],
                        file_name=f"report_{record['image_name']}.md",
                        mime="text/markdown",
                    )

                with tab_rework:
                    st.write(f"### Disposition: **{advisory_result['disposition']}**")
                    st.write(f"**Severity:** `{advisory_result['severity']}`")
                    st.markdown("#### Actionable Operator Steps:")
                    for idx, step in enumerate(advisory_result["rework_instructions"], 1):
                        st.checkbox(f"{step}", key=f"step_{idx}_{unique_id}")

                with tab_prompt:
                    st.markdown("**Structured Vision Output Passed to Llama:**")
                    st.json(record)
                    st.markdown("**Llama 3 Instruct Prompt Template:**")
                    st.code(advisor.build_llama_prompt(record), language="markdown")

                # Defect Log CSV
                if record["detections"]:
                    df = pd.DataFrame(record["detections"])
                    st.download_button(
                        "📥 Download Detection Log (CSV)",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=f"inspection_{record['image_name']}.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Inspection failed: {exc}")
                st.exception(exc)
else:
    st.info("Upload an image or choose a test-set image to begin.")

st.markdown("---")
st.caption(
    "EdgeInspect AI — Arm Bharat AI-SoC Challenge 2026–27 (PS5). "
    "Combines YOLOv8 Nano edge defect detection with on-device Llama reasoning via ExecuTorch on Arm CPU."
)
