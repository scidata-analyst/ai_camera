"""Detect Rockchip NPU character devices and DRM render-node drivers.

On some Rockchip Ubuntu images, RKNPU is exposed as a DRM node such as
/dev/dri/renderD129 instead of /dev/rknpu. Read the sysfs driver link so the
health endpoint does not mistake every render node for a GPU.
"""
import glob
import os

import numpy as np

RENDER_GLOB = "/dev/dri/renderD*"
RKNPU_GLOB = "/dev/rknpu*"
# user er deoa explicit path (compose e mount hoy)
EXPLICIT_RENDER = "/dev/dri/renderD129"


def dri_devices() -> list:
    return sorted(glob.glob(RENDER_GLOB))


def rknpu_devices() -> list:
    return sorted(glob.glob(RKNPU_GLOB))


def dri_driver_map() -> dict:
    """Map each DRM render node to its kernel driver when sysfs exposes it."""
    result = {}
    for path in dri_devices():
        node = os.path.basename(path)
        driver_link = f"/sys/class/drm/{node}/device/driver"
        if os.path.exists(driver_link):
            result[path] = os.path.basename(os.path.realpath(driver_link))
        else:
            result[path] = None
    return result


def hw_summary() -> dict:
    dri = dri_devices()
    rknpu = rknpu_devices()
    drivers = dri_driver_map()
    npu_dri = sorted(path for path, driver in drivers.items()
                     if driver and "rknpu" in driver.lower())
    gpu_drivers = ("mali", "panfrost")
    return {
        "render_explicit": EXPLICIT_RENDER,
        "render_exists": os.path.exists(EXPLICIT_RENDER),
        "dri": dri,
        "dri_drivers": drivers,
        "npu_dri": npu_dri,
        "rknpu": rknpu,
        "has_gpu": any(driver and any(g in driver.lower() for g in gpu_drivers)
                        for driver in drivers.values()),
        "has_npu": bool(rknpu or npu_dri),
    }


def rknn_infer(rknn, blob, layout: str, dtype: str | None = None):
    """Run RKNNLite.inference with an explicit input layout.

    Lite2 defaults data_format to NHWC when it is omitted. That silently
    misreads an NCHW float blob as (N=1, H=3, W=H, C=W) and produces garbage
    detections (typically everything becomes COCO 'person'). Always state the
    real layout of the array we pass. Older runtimes reject the kwargs, so
    fall back to the historical call in that case.
    """
    kw = {"inputs": [blob], "data_format": layout}
    if dtype:
        kw["data_type"] = dtype
    try:
        return rknn.inference(**kw)
    except TypeError:
        return rknn.inference(inputs=[blob])
    except Exception as exc:
        # Kono board-e runtime NCHW buffer accept na korle NHWC try koro.
        if layout == "nchw" and blob.ndim == 4 and blob.shape[1] in (1, 3, 4):
            hwc = np.ascontiguousarray(
                np.transpose(blob[0], (1, 2, 3, 0))[np.newaxis, ...])
            retry = {"inputs": [hwc], "data_format": "nhwc"}
            if dtype:
                retry["data_type"] = dtype
            print(f"[rknn] NCHW inference failed ({exc}); retry NHWC")
            return rknn.inference(**retry)
        raise


def rknn_input_summary(rknn) -> dict:
    """Best-effort model input descriptor, used for startup diagnostics."""
    try:
        details = rknn.get_input_details()
        if not details:
            return {}
        d = details[0]
        return {
            "shape": tuple(int(v) for v in d.get("shape", ())),
            "dtype": str(d.get("dtype", "")),
            "layout": str(d.get("layout", "")).lower(),
        }
    except Exception:
        return {}


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
