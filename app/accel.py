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


def _runtime_input_layout(rknn, index: int = 0) -> tuple[str, tuple]:
    """Read the layout the RKNN runtime was actually compiled for."""
    try:
        details = rknn.get_input_details()
        if details and index < len(details):
            d = details[index]
            return (str(d.get("layout", "")).lower(),
                    tuple(int(v) for v in d.get("shape", ())))
    except Exception:
        pass
    return "", ()


def _as_4d(blob: np.ndarray, layout: str) -> np.ndarray:
    """RKNN requires a batch dimension; a bare HWC/NCHW array is rejected."""
    if blob.ndim == 3:
        return np.ascontiguousarray(blob[np.newaxis, ...])
    return blob


def _reorder(blob: np.ndarray, src: str, dst: str) -> np.ndarray:
    if not src or not dst or src == dst:
        return blob
    if src == "nchw" and dst == "nhwc":
        return np.ascontiguousarray(np.transpose(blob, (0, 2, 3, 1)))
    if src == "nhwc" and dst == "nchw":
        return np.ascontiguousarray(np.transpose(blob, (0, 3, 1, 2)))
    return blob


def rknn_infer(rknn, blob, layout: str, dtype: str | None = None):
    """Run RKNNLite.inference with an explicit 4D input layout.

    Lite2 defaults data_format to NHWC when it is omitted. That silently
    misreads an NCHW float blob as (N=1, H=3, W=H, C=W) and produces garbage
    detections (typically everything becomes COCO 'person'). These RK3576
    models are compiled for NHWC, so normalize every call to the runtime's
    reported layout and never send a 3D array.
    """
    blob = _as_4d(blob, layout)
    runtime_layout, runtime_shape = _runtime_input_layout(rknn)
    if runtime_layout in ("nchw", "nhwc"):
        blob = _reorder(blob, layout, runtime_layout)
        layout = runtime_layout
    elif len(runtime_shape) == 4:
        # Some Lite releases omit 'layout' but still expose the shape.
        if runtime_shape[-1] in (1, 3, 4) and blob.shape[1] in (1, 3, 4) \
                and blob.shape[-1] not in (1, 3, 4):
            blob = _reorder(blob, "nchw", "nhwc")
            layout = "nhwc"
        elif runtime_shape[1] in (1, 3, 4) and blob.shape[-1] in (1, 3, 4) \
                and blob.shape[1] not in (1, 3, 4):
            blob = _reorder(blob, "nhwc", "nchw")
            layout = "nchw"

    kw = {"inputs": [blob], "data_format": layout}
    if dtype:
        kw["data_type"] = dtype
    try:
        return rknn.inference(**kw)
    except TypeError:
        return rknn.inference(inputs=[blob])
    except Exception as exc:
        # get_input_details() kono samoy compiled layout na janale other layout
        # ekbar try koro. Batch dim thaklei 4D requirement fulfillment hoy.
        other = "nhwc" if layout == "nchw" else "nchw"
        if blob.ndim == 4 and blob.shape[0] == 1:
            retry_blob = _reorder(blob, layout, other)
            retry = {"inputs": [retry_blob], "data_format": other}
            if dtype:
                retry["data_type"] = dtype
            try:
                return rknn.inference(**retry)
            except Exception:
                pass
        raise RuntimeError(
            f"RKNN inference failed ({exc}); tried {layout} and {other}") from exc


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


def rknn_load(path: str, label: str = "model") -> tuple:
    """Load + init an RKNN model on NPU. No ONNX/CPU fallback, ever."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"[{label}] RKNN model missing: {path} — CPU fallback disabled")
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    ret = rknn.load_rknn(path)
    if ret != 0:
        raise RuntimeError(f"[{label}] load_rknn failed: ret={ret}")
    core = os.getenv("RKNPU_CORE", "AUTO")
    kw = {}
    if core.upper() not in ("AUTO", ""):
        try:
            kw["core_mask"] = int(core)
        except ValueError:
            pass
    ret = rknn.init_runtime(**kw) if kw else rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"[{label}] init_runtime failed: ret={ret}")
    return rknn, core
