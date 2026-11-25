from __future__ import annotations

import glob
import os

import numpy as np

from ...config.constants import EXPLICIT_RENDER, RENDER_GLOB, RKNPU_GLOB


def dri_devices() -> list:

    return sorted(glob.glob(RENDER_GLOB))


def rknpu_devices() -> list:

    return sorted(glob.glob(RKNPU_GLOB))


def dri_driver_map() -> dict:
    result = {}

    for path in dri_devices():
        node        = os.path.basename(path)
        driver_link = f"/sys/class/drm/{node}/device/driver"

        if os.path.exists(driver_link):
            result[path] = os.path.basename(os.path.realpath(driver_link))
        else:
            result[path] = None

    return result


def hw_summary() -> dict:
    dri         = dri_devices()
    rknpu       = rknpu_devices()
    drivers     = dri_driver_map()
    npu_dri     = sorted(path for path, driver in drivers.items() if driver and "rknpu" in driver.lower())
    gpu_drivers = ("mali", "panfrost")

    return {"render_explicit": EXPLICIT_RENDER, "render_exists": os.path.exists(EXPLICIT_RENDER), "dri": dri, "dri_drivers": drivers, "npu_dri": npu_dri, "rknpu": rknpu, "has_gpu": any(driver and any(g in driver.lower() for g in gpu_drivers) for driver in drivers.values()), "has_npu": bool(rknpu or npu_dri),}


def _runtime_input_layout(rknn, index: int = 0) -> tuple[str, tuple]:

    try:
        details = rknn.get_input_details()

        if details and index < len(details):
            detail = details[index]

            return (str(detail.get("layout", "")).lower(), tuple(int(v) for v in detail.get("shape", ())))
    except Exception:
        pass


    return "", ()


def _as_4d(blob: np.ndarray, layout: str) -> np.ndarray:

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
    blob = _as_4d(blob, layout)
    runtime_layout, runtime_shape = _runtime_input_layout(rknn)

    if runtime_layout in ("nchw", "nhwc"):
        blob   = _reorder(blob, layout, runtime_layout)
        layout = runtime_layout
    elif len(runtime_shape) == 4:

        if runtime_shape[- 1] in (1, 3, 4) and blob.shape[1] in (1, 3, 4) and blob.shape[- 1] not in (1, 3, 4):
            blob   = _reorder(blob, "nchw", "nhwc")
            layout = "nhwc"
        elif runtime_shape[1] in (1, 3, 4) and blob.shape[- 1] in (1, 3, 4) and blob.shape[1] not in (1, 3, 4):
            blob   = _reorder(blob, "nhwc", "nchw")
            layout = "nchw"

    kwargs = {"inputs": [blob], "data_format": layout}

    if dtype:
        kwargs["data_type"] = dtype

    try:

        return rknn.inference(**kwargs)
    except TypeError:

        return rknn.inference(inputs=[blob])
    except Exception as exc:
        other = "nhwc" if layout == "nchw" else "nchw"

        if blob.ndim == 4 and blob.shape[0] == 1:
            retry_blob = _reorder(blob, layout, other)
            retry      = {"inputs": [retry_blob], "data_format": other}

            if dtype:
                retry["data_type"] = dtype

            try:

                return rknn.inference(**retry)
            except Exception:
                pass

        raise RuntimeError(
            f"RKNN inference failed ({exc}); tried {layout} and {other}") from exc



def rknn_input_summary(rknn) -> dict:

    try:
        details = rknn.get_input_details()

        if not details:

            return {}
        detail = details[0]

        return {"shape": tuple(int(v) for v in detail.get("shape", ())), "dtype": str(detail.get("dtype", "")), "layout": str(detail.get("layout", "")).lower(),}
    except Exception:

        return {}



def rknn_load(path: str, label: str = "model") -> tuple:

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"[{label}] RKNN model missing: {path} — CPU fallback disabled")
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    ret  = rknn.load_rknn(path)

    if ret != 0:
        raise RuntimeError(f"[{label}] load_rknn failed: ret={ret}")
    core   = os.getenv("RKNPU_CORE", "AUTO")
    kwargs = {}

    if core.upper() not in ("AUTO", ""):

        try:
            kwargs["core_mask"] = int(core)
        except ValueError:
            pass

    ret = rknn.init_runtime(**kwargs) if kwargs else rknn.init_runtime()

    if ret != 0:
        raise RuntimeError(f"[{label}] init_runtime failed: ret={ret}")

    return rknn, core
