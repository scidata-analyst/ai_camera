"""Manually convert the RetinaFace and InsightFace GenderAge ONNX models.

Run on an x86_64 Linux machine with RKNN-Toolkit2 installed:
    python tools/export_face_models_rknn.py --target rk3576

Compose does not invoke this exporter. Both models are exported as FP32 to
avoid introducing INT8 calibration error into face/attribute predictions.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def convert(model: Path, output: Path, target: str,
            mean_values, std_values, input_name: str | None = None,
            input_size: tuple[int, ...] | None = None) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        rknn.config(
            target_platform=target,
            mean_values=mean_values,
            std_values=std_values,
        )
        if input_name and input_size:
            # GenderAge ONNX leaves the batch dimension dynamic. Its runtime
            # contract is a single NCHW face crop, so fix batch=1 for RKNN.
            ret = rknn.load_onnx(
                model=str(model), inputs=[input_name],
                input_size_list=[list(input_size)],
            )
        else:
            ret = rknn.load_onnx(model=str(model))
        if ret != 0:
            raise RuntimeError(f"load_onnx failed for {model}: {ret}")
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="face-rknn-build-") as tmp:
            os.chdir(tmp)
            try:
                ret = rknn.build(do_quantization=False)
                if ret != 0:
                    raise RuntimeError(f"build failed for {model}: {ret}")
                ret = rknn.export_rknn(str(output))
                if ret != 0:
                    raise RuntimeError(f"export_rknn failed for {output}: {ret}")
            finally:
                os.chdir(old_cwd)
        print(f"[export] saved {output} ({output.stat().st_size / 1e6:.2f} MB)")
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="rk3576")
    args = parser.parse_args()

    retinaface = ROOT / "models/RetinaFace_mobile320.onnx"
    genderage = ROOT / "models/genderage.onnx"
    if not retinaface.is_file() or not genderage.is_file():
        raise SystemExit("Missing ONNX files in models/. Download them first.")

    # Match the official RKNN Model Zoo RetinaFace conversion contract:
    # the app supplies a 320x320 RGB uint8 image.
    convert(
        retinaface,
        ROOT / "models/RetinaFace_mobile320-rk3576.rknn",
        args.target, [[104, 117, 123]], [[1, 1, 1]],
    )

    # GenderAge's official ONNX graph already contains pixel normalization;
    # the app supplies its aligned 96x96 RGB crop as raw 0..255 float32.
    convert(
        genderage,
        ROOT / "models/genderage-rk3576.rknn",
        args.target, [[0, 0, 0]], [[1, 1, 1]],
        input_name="data", input_size=(1, 3, 96, 96),
    )


if __name__ == "__main__":
    main()
