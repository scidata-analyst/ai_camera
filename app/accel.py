"""Hardware accel helper: NPU (/dev/rknpu) + GPU (/dev/dri/renderD129) detect.

RK3576 te:
  /dev/rknpu            = NPU (RKNNLite YOLO)
  /dev/dri/renderD129  = Mali GPU (OpenCL: YuNet/WHENet offload)
Board e renderD128/D129 naam bodlate pare, tai /dev/dri/* scan kori.
x86 dev PC te renderD128 (Intel iGPU) thakleo same code safe fallback hoy.
"""
import glob
import os

RENDER_GLOB = "/dev/dri/renderD*"
RKNPU_GLOB = "/dev/rknpu*"
# user er deoa explicit path (compose e mount hoy)
EXPLICIT_RENDER = "/dev/dri/renderD129"


def dri_devices() -> list:
    return sorted(glob.glob(RENDER_GLOB))


def rknpu_devices() -> list:
    return sorted(glob.glob(RKNPU_GLOB))


def hw_summary() -> dict:
    dri = dri_devices()
    rknpu = rknpu_devices()
    return {
        "render_explicit": EXPLICIT_RENDER,
        "render_exists": os.path.exists(EXPLICIT_RENDER),
        "dri": dri,
        "rknpu": rknpu,
        "has_gpu": len(dri) > 0,
        "has_npu": len(rknpu) > 0,
    }


def opencl_available() -> bool:
    """OpenCL runtime sotti kaj kore kina (device thaklei hobe na)."""
    if os.getenv("OPENCV_OPENCL", "auto").lower() == "off":
        return False
    try:
        import cv2
        return bool(cv2.ocl.haveOpenCL())
    except Exception:
        return False


def try_enable_opencl(on: bool = True) -> bool:
    try:
        import cv2
        if not cv2.ocl.haveOpenCL():
            return False
        cv2.ocl.setUseOpenCL(bool(on))
        return bool(cv2.ocl.useOpenCL()) if on else False
    except Exception:
        return False


def pick_ort_providers(want: str = "auto") -> list:
    """ONNXRuntime provider list. want=auto hole GPU provider try + CPU fallback."""
    want = (want or "auto").lower()
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    if want not in ("auto", ""):
        # explicit: "OpenVINOExecutionProvider,CPUExecutionProvider"
        asked = [p.strip() for p in want.split(",") if p.strip()]
        return [p for p in asked if p in avail] or ["CPUExecutionProvider"]
    # auto: GPU-ish provider thakle age, sobses e CPU (crash-proof order)
    order = ["OpenVINOExecutionProvider", "VitisAIExecutionProvider",
             "OpenCLExecutionProvider", "CPUExecutionProvider"]
    picked = [p for p in order if p in avail]
    return picked or ["CPUExecutionProvider"]
