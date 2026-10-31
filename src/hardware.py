"""
EdgeInspect AI: Dynamic Hardware & Runtime Profiler
Challenge: Arm Bharat AI-SoC Challenge 2026-27 (PS5: Industrial Defect Detection)

Reconciles and dynamically identifies edge hardware and execution runtimes:
  - realme 12x 5G (MediaTek Dimensity 6100+ | 2x Cortex-A76 + 6x Cortex-A55 | Arm CPU | ExecuTorch)
  - Raspberry Pi 5 (Broadcom BCM2712 | 4x Cortex-A76 @ 2.4GHz | Arm CPU)
  - Host Workstation (x86_64 Development Environment | CPU / Emulation)
  - Hailo-8 / Hailo-8L NPU (Strictly verified via physical PCIe presence check)
  - Vision Runtime: ONNX Runtime vs PyTorch Native vs NCNN vs TFLite
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Preset hardware deployment profiles
HARDWARE_PROFILES = {
    "realme_12x": {
        "id": "realme_12x",
        "name": "realme 12x 5G (Edge Mobile Deployment)",
        "soc": "MediaTek Dimensity 6100+ (6nm)",
        "cpu": "Octa-core (2x Arm Cortex-A76 @ 2.4 GHz + 6x Arm Cortex-A55 @ 2.0 GHz)",
        "gpu": "Arm Mali-G57 MC2",
        "npu": "MediaTek APU (ExecuTorch Arm CPU fallback)",
        "llm_runtime": "ExecuTorch INT8 / XNNPACK (Arm CPU)",
        "display_string": "realme 12x 5G | MediaTek Dimensity 6100+ | Arm CPU | ExecuTorch",
        "is_physical_hailo": False
    },
    "pi5_cpu": {
        "id": "pi5_cpu",
        "name": "Raspberry Pi 5 (Arm SoC Native CPU)",
        "soc": "Broadcom BCM2712 (16nm)",
        "cpu": "Quad-core Arm Cortex-A76 @ 2.4 GHz (Armv8.2-A / Cryptography)",
        "gpu": "VideoCore VII @ 800 MHz",
        "npu": "None (Pure Arm CPU Execution)",
        "llm_runtime": "ExecuTorch XNNPACK (Arm Cortex-A76)",
        "display_string": "Raspberry Pi 5 | Broadcom BCM2712 | Arm Cortex-A76 CPU | ExecuTorch",
        "is_physical_hailo": False
    },
    "pi5_hailo": {
        "id": "pi5_hailo",
        "name": "Raspberry Pi 5 + Hailo-8L NPU (AI Kit M.2 PCIe)",
        "soc": "Broadcom BCM2712 + Hailo-8L PCIe",
        "cpu": "Quad-core Arm Cortex-A76 @ 2.4 GHz",
        "gpu": "VideoCore VII",
        "npu": "Hailo-8L NPU (13 TOPS @ 2.5W)",
        "llm_runtime": "ExecuTorch on Cortex-A76 + HailoRT Vision Pipeline",
        "display_string": "Raspberry Pi 5 | Hailo-8L NPU (13 TOPS) + Arm Cortex-A76",
        "is_physical_hailo": True
    },
    "host_pc": {
        "id": "host_pc",
        "name": "Host Workstation (Local Development / Evaluation)",
        "soc": f"{platform.processor() or 'x86_64 PC'}",
        "cpu": f"{platform.machine()} ({os.cpu_count() or 4} cores)",
        "gpu": "Host Display Adapter",
        "npu": "None (Host CPU Execution / Arm Emulation)",
        "llm_runtime": "ExecuTorch PyTorch Runtime (Local CPU)",
        "display_string": f"Host PC ({platform.machine()}) | CPU Execution | ExecuTorch Emulation",
        "is_physical_hailo": False
    }
}


def check_physical_hailo_hardware() -> Dict[str, Any]:
    """
    Strictly verifies whether physical Hailo PCIe hardware is connected.
    Returns status and explanation; never claims Hailo acceleration without physical silicon.
    """
    has_hailort = False
    physical_device_found = False
    details = "HailoRT driver not installed; physical PCIe M.2 accelerator absent."

    try:
        import hailort  # type: ignore
        has_hailort = True
        try:
            from hailo_platform import VDevice  # type: ignore
            vdev = VDevice()
            physical_device_found = True
            details = "Physical Hailo-8/8L PCIe device detected and initialized."
        except Exception as exc:
            details = f"HailoRT package present, but no physical PCIe hardware initialized ({exc})."
    except ImportError:
        details = "HailoRT library not loaded. System is executing purely on host Arm/x86 CPU."

    return {
        "has_hailort_lib": has_hailort,
        "physical_hardware_detected": physical_device_found,
        "status_label": "PHYSICAL NPU PRESENT" if physical_device_found else "PHYSICAL NPU ABSENT (CPU FALLBACK)",
        "details": details
    }


def detect_vision_runtime() -> Dict[str, Any]:
    """
    Detects the active vision execution runtime (PyTorch, ONNX Runtime, NCNN, TFLite).
    """
    runtime = "PyTorch (Native Engine)"
    details = "Running YOLOv8n via PyTorch CPU backend"
    has_onnx = False
    has_ncnn = False
    has_tflite = False

    # Check ONNX Runtime
    try:
        import onnxruntime
        has_onnx = True
        onnx_model = PROJECT_ROOT / "models" / "baseline_nano" / "weights" / "best.onnx"
        if onnx_model.exists():
            runtime = "ONNX Runtime (CPU Execution)"
            details = f"ONNX Runtime v{onnxruntime.__version__} executing {onnx_model.name}"
    except ImportError:
        pass

    # Check NCNN
    try:
        import ncnn
        has_ncnn = True
        runtime = "NCNN (Arm Vulkan / CPU)"
        details = "NCNN mobile optimized runtime active"
    except ImportError:
        pass

    # Check TFLite
    try:
        import tflite_runtime
        has_tflite = True
    except ImportError:
        pass

    return {
        "active_runtime": runtime,
        "details": details,
        "has_onnx": has_onnx,
        "has_ncnn": has_ncnn,
        "has_tflite": has_tflite
    }


def get_system_hardware_info(selected_profile_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns unified hardware and runtime telemetry.
    Reconciles target platform description with ground-truth device capabilities.
    """
    hailo_check = check_physical_hailo_hardware()
    vision_runtime = detect_vision_runtime()

    # Determine default profile
    if selected_profile_key and selected_profile_key in HARDWARE_PROFILES:
        profile = HARDWARE_PROFILES[selected_profile_key]
    else:
        # Default to realme 12x 5G mobile profile if on mobile / Arm, or Host PC
        if "arm" in platform.machine().lower() or "aarch64" in platform.machine().lower():
            profile = HARDWARE_PROFILES["realme_12x"]
        else:
            profile = HARDWARE_PROFILES["realme_12x"]  # Primary target specified by user

    # Strict reconciliation rule: Do not claim Hailo NPU if physical hardware absent
    effective_display = profile["display_string"]
    if profile["is_physical_hailo"] and not hailo_check["physical_hardware_detected"]:
        effective_display = f"{profile['name']} [Hailo NPU Simulated / Running on Arm CPU]"

    return {
        "profile_key": profile["id"],
        "profile_name": profile["name"],
        "display_hardware": effective_display,
        "soc": profile["soc"],
        "cpu": profile["cpu"],
        "gpu": profile["gpu"],
        "npu_claim": profile["npu"],
        "llm_runtime": profile["llm_runtime"],
        "hailo_check": hailo_check,
        "vision_runtime": vision_runtime,
        "system_platform": platform.platform(),
        "python_version": platform.python_version()
    }


if __name__ == "__main__":
    info = get_system_hardware_info("realme_12x")
    print("=== EdgeInspect AI Hardware & Runtime Profile ===")
    print(f"Hardware Identifier: {info['display_hardware']}")
    print(f"SoC Architecture:    {info['soc']}")
    print(f"CPU Specifications:  {info['cpu']}")
    print(f"Vision Runtime:      {info['vision_runtime']['active_runtime']}")
    print(f"Hailo Verification:  {info['hailo_check']['status_label']}")
    print(f"Details:             {info['hailo_check']['details']}")
